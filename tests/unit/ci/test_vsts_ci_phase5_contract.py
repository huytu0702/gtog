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



def test_validate_stage_keeps_security_scanning_and_quality_checks():
    stage = _stage_map()["Validate"]
    jobs = stage["jobs"]
    assert jobs

    task_names = _iter_step_values(jobs, "task")
    scripts = _iter_step_values(jobs, "script")
    combined_scripts = "\n".join(scripts)

    assert "CredScan@3" in task_names
    assert "ComponentGovernanceComponentDetection@0" in task_names
    assert "PublishSecurityAnalysisLogs@3" in task_names
    assert "uv run poe check" in combined_scripts
    assert "pytest ./backend/tests/unit ./tests/unit" in combined_scripts



def test_stage_dependencies_form_release_gate_chain():
    stages = _stage_map()

    assert stages["BuildImages"]["dependsOn"] == ["Validate"]
    assert stages["DeployStaging"]["dependsOn"] == ["BuildImages"]
    assert stages["SmokeStaging"]["dependsOn"] == ["DeployStaging"]
    assert stages["ManualApproval"]["dependsOn"] == ["SmokeStaging"]
    assert stages["DeployProduction"]["dependsOn"] == ["ManualApproval"]



def test_smoke_stage_publishes_required_evidence_artifacts():
    stage = _stage_map()["SmokeStaging"]
    publish_artifacts: list[str] = []
    for job in stage["jobs"]:
        for step in job.get("steps", []):
            artifact_name = step.get("artifact")
            if isinstance(artifact_name, str):
                publish_artifacts.append(artifact_name)

    assert "$(smokeStagingArtifactName)" in publish_artifacts
    assert "phase3-auth-origin-validation" in publish_artifacts



def test_production_stage_uses_built_artifacts_and_not_rebuilds():
    stage = _stage_map()["DeployProduction"]
    task_names = _iter_step_values(stage["jobs"], "task")
    scripts = _iter_step_values(stage["jobs"], "script")
    combined_scripts = "\n".join(scripts).lower()

    assert "DownloadPipelineArtifact@2" in task_names
    assert "docker build" not in combined_scripts
