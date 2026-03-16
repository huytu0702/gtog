from __future__ import annotations

from pathlib import Path

import yaml


PIPELINE_PATH = Path(__file__).resolve().parents[3] / ".vsts-ci.yml"
REQUIRED_STAGES = [
    "Validate",
    "BuildImages",
    "DeployStaging",
    "SmokeStaging",
    "ManualApproval",
    "DeployProduction",
    "SmokeProductionCanary",
    "RollbackProductionCanary",
    "ObserveProductionCanary",
    "ManualApprovalCanary",
    "PromoteProductionFull",
    "SmokeProductionFull",
]


def _load_pipeline() -> dict:
    return yaml.safe_load(PIPELINE_PATH.read_text(encoding="utf-8"))


def _stage_map() -> dict[str, dict]:
    pipeline = _load_pipeline()
    return {stage["stage"]: stage for stage in pipeline["stages"]}


def _stage_list() -> list[dict]:
    return _load_pipeline()["stages"]


def _iter_step_values(jobs: list[dict], key: str) -> list[str]:
    values: list[str] = []
    for job in jobs:
        for step in job.get("steps", []):
            value = step.get(key)
            if isinstance(value, str):
                values.append(value)
    return values


def test_pipeline_contains_phase5_stage_sequence():
    stage_names = [stage["stage"] for stage in _stage_list()]

    assert stage_names[: len(REQUIRED_STAGES)] == REQUIRED_STAGES


def test_validate_stage_keeps_current_quality_checks_and_disables_unavailable_compliance_tasks():
    stage = _stage_map()["Validate"]
    jobs = {job["job"]: job for job in stage["jobs"]}

    quality_job = jobs["quality_checks"]
    scripts = _iter_step_values([quality_job], "script")
    combined_scripts = "\n".join(scripts)

    assert "uv sync --dev" in combined_scripts
    assert "uv run poe check" in combined_scripts
    assert "pytest ./backend/tests/unit ./tests/unit" in combined_scripts

    compliance_job = jobs["compliance"]
    assert compliance_job["condition"] is False
    assert compliance_job["pool"]["vmImage"] == "windows-latest"


def test_stage_dependencies_form_release_gate_chain():
    stages = _stage_map()

    assert stages["BuildImages"]["dependsOn"] == ["Validate"]
    assert stages["DeployStaging"]["dependsOn"] == ["BuildImages"]
    assert stages["SmokeStaging"]["dependsOn"] == ["DeployStaging"]
    assert stages["ManualApproval"]["dependsOn"] == ["SmokeStaging"]
    assert stages["DeployProduction"]["dependsOn"] == ["ManualApproval"]
    assert stages["SmokeProductionCanary"]["dependsOn"] == ["DeployProduction"]
    assert stages["RollbackProductionCanary"]["dependsOn"] == ["SmokeProductionCanary"]
    assert stages["ObserveProductionCanary"]["dependsOn"] == ["SmokeProductionCanary"]
    assert stages["ManualApprovalCanary"]["dependsOn"] == ["ObserveProductionCanary"]
    assert stages["PromoteProductionFull"]["dependsOn"] == ["ManualApprovalCanary"]
    assert stages["SmokeProductionFull"]["dependsOn"] == ["PromoteProductionFull"]


def test_smoke_stage_publishes_required_evidence_artifacts():
    stage = _stage_map()["SmokeStaging"]
    publish_artifacts: list[str] = []
    download_steps: list[dict] = []
    run_step_index: int | None = None
    run_step_env: dict | None = None

    for job in stage["jobs"]:
        for step_index, step in enumerate(job.get("steps", [])):
            artifact_name = step.get("artifact")
            if isinstance(artifact_name, str):
                publish_artifacts.append(artifact_name)
            if step.get("task") == "DownloadPipelineArtifact@2":
                download_steps.append(step)
            if step.get("displayName") == "Run staging smoke gates":
                run_step_index = step_index
                run_step_env = step.get("env", {})

    assert "$(smokeStagingArtifactName)" in publish_artifacts
    assert "$(phase3ValidationArtifactName)" in publish_artifacts
    assert run_step_index is not None
    assert run_step_env is not None
    assert run_step_env["APP_BASE_URL"] == "https://$(stagingAppPublicHostname)"
    assert run_step_env["APP_PUBLIC_HOSTNAME"] == "$(stagingAppPublicHostname)"
    assert "EXPECTED_CLIENT_ID" not in run_step_env
    assert any(
        step.get("inputs", {}).get("artifactName") == "$(stagingDeployArtifactName)"
        and step.get("inputs", {}).get("targetPath") == "$(Pipeline.Workspace)/deploy-staging-log"
        for step in download_steps
    )
    assert any(
        job_step.get("task") == "DownloadPipelineArtifact@2"
        and job_step.get("inputs", {}).get("artifactName") == "$(stagingDeployArtifactName)"
        and step_index < run_step_index
        for job in stage["jobs"]
        for step_index, job_step in enumerate(job.get("steps", []))
    )


def test_production_stage_uses_built_artifacts_and_not_rebuilds():
    stage = _stage_map()["DeployProduction"]
    task_names = _iter_step_values(stage["jobs"], "task")
    scripts = _iter_step_values(stage["jobs"], "script")
    combined_scripts = "\n".join(scripts).lower()

    assert "DownloadPipelineArtifact@2" in task_names
    assert "docker build" not in combined_scripts


def test_phase6_production_rollout_stages_publish_canary_and_full_evidence():
    stages = _stage_map()

    canary_artifacts = [
        step.get("artifact")
        for job in stages["SmokeProductionCanary"]["jobs"]
        for step in job.get("steps", [])
        if isinstance(step.get("artifact"), str)
    ]
    observation_artifacts = [
        step.get("artifact")
        for job in stages["ObserveProductionCanary"]["jobs"]
        for step in job.get("steps", [])
        if isinstance(step.get("artifact"), str)
    ]
    rollback_artifacts = [
        step.get("artifact")
        for job in stages["RollbackProductionCanary"]["jobs"]
        for step in job.get("steps", [])
        if isinstance(step.get("artifact"), str)
    ]
    full_artifacts = [
        step.get("artifact")
        for job in stages["SmokeProductionFull"]["jobs"]
        for step in job.get("steps", [])
        if isinstance(step.get("artifact"), str)
    ]

    assert "$(smokeProductionCanaryArtifactName)" in canary_artifacts
    assert "$(phase3ProductionCanaryValidationArtifactName)" in canary_artifacts
    assert "$(productionCanaryObservationArtifactName)" in observation_artifacts
    assert "$(productionRolloutStateArtifactName)" in observation_artifacts
    assert "$(productionRollbackArtifactName)" in rollback_artifacts
    assert "$(smokeProductionFullArtifactName)" in full_artifacts
    assert "$(phase3ProductionFullValidationArtifactName)" in full_artifacts


def test_phase6_rollout_reuses_release_artifacts_without_rebuilds():
    deploy_stage = _stage_map()["DeployProduction"]
    promote_stage = _stage_map()["PromoteProductionFull"]
    rollback_stage = _stage_map()["RollbackProductionCanary"]

    deploy_scripts = "\n".join(_iter_step_values(deploy_stage["jobs"], "script")).lower()
    promote_scripts = "\n".join(_iter_step_values(promote_stage["jobs"], "script")).lower()
    rollback_scripts = "\n".join(_iter_step_values(rollback_stage["jobs"], "script")).lower()

    assert "docker build" not in deploy_scripts
    assert "docker build" not in promote_scripts
    assert "docker build" not in rollback_scripts
    assert 'cp "$(pipeline.workspace)/production-rollout-state/rollout-state.json" "$(build.artifactstagingdirectory)/production-rollout-state/rollout-state.json"' in promote_scripts
