from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "validate.yml"
RELEASE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
REQUIRED_RELEASE_JOBS = [
    "validate_release",
    "build_images",
    "deploy_staging",
    "smoke_staging",
    "await_prod_approval",
    "deploy_production_canary",
    "smoke_production_canary",
    "rollback_production_canary",
    "observe_production_canary",
    "await_full_promotion",
    "promote_production_full",
    "smoke_production_full",
]
EXPECTED_RELEASE_ARTIFACTS = {
    "deploy-production-log",
    "deploy-staging-log",
    "image-metadata",
    "phase3-auth-origin-validation",
    "phase3-auth-origin-validation-production-canary",
    "phase3-auth-origin-validation-production-full",
    "production-canary-observation",
    "production-rollback-log",
    "production-rollout-state",
    "smoke-production-canary-report",
    "smoke-production-full-report",
    "smoke-staging-report",
    "validate-stage-report",
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflow_jobs(path: Path) -> dict[str, dict]:
    return _load_yaml(path)["jobs"]


def _iter_run_steps(job: dict) -> list[str]:
    return [step["run"] for step in job.get("steps", []) if isinstance(step.get("run"), str)]


def _iter_upload_artifacts(jobs: dict[str, dict]) -> set[str]:
    artifacts: set[str] = set()
    for job in jobs.values():
        for step in job.get("steps", []):
            if step.get("uses") == "actions/upload-artifact@v4":
                name = step.get("with", {}).get("name")
                if isinstance(name, str):
                    artifacts.add(name)
    return artifacts


def test_validate_workflow_runs_on_pull_requests_and_main_pushes():
    workflow = _load_yaml(VALIDATE_WORKFLOW_PATH)

    assert VALIDATE_WORKFLOW_PATH.exists()
    assert workflow["on"]["pull_request"] == {}
    assert workflow["on"]["push"]["branches"] == ["main"]



def test_validate_workflow_reuses_repo_checks_and_frontend_build_validation():
    jobs = _workflow_jobs(VALIDATE_WORKFLOW_PATH)
    validate_job = jobs["quality_checks"]
    scripts = "\n".join(_iter_run_steps(validate_job))

    assert "uv sync --dev" in scripts
    assert "uv run poe check" in scripts
    assert "uv run pytest ./backend/tests/unit ./tests/unit" in scripts
    assert "npm ci" in scripts
    assert "npm run build" in scripts
    assert "actions/upload-artifact@v4" in [step.get("uses") for step in validate_job.get("steps", [])]



def test_release_workflow_contains_expected_job_chain():
    jobs = _workflow_jobs(RELEASE_WORKFLOW_PATH)

    assert RELEASE_WORKFLOW_PATH.exists()
    assert list(jobs)[: len(REQUIRED_RELEASE_JOBS)] == REQUIRED_RELEASE_JOBS
    assert jobs["build_images"]["needs"] == ["validate_release"]
    assert jobs["deploy_staging"]["needs"] == ["build_images"]
    assert jobs["smoke_staging"]["needs"] == ["deploy_staging"]
    assert jobs["await_prod_approval"]["needs"] == ["smoke_staging"]
    assert jobs["deploy_production_canary"]["needs"] == ["await_prod_approval"]
    assert jobs["smoke_production_canary"]["needs"] == ["deploy_production_canary"]
    assert jobs["rollback_production_canary"]["needs"] == ["smoke_production_canary"]
    assert jobs["observe_production_canary"]["needs"] == ["smoke_production_canary"]
    assert jobs["await_full_promotion"]["needs"] == ["observe_production_canary"]
    assert jobs["promote_production_full"]["needs"] == ["await_full_promotion"]
    assert jobs["smoke_production_full"]["needs"] == ["promote_production_full"]



def test_release_workflow_preserves_artifact_contract_and_script_reuse():
    jobs = _workflow_jobs(RELEASE_WORKFLOW_PATH)
    all_scripts = "\n".join(
        run_step
        for job in jobs.values()
        for run_step in _iter_run_steps(job)
    )

    assert EXPECTED_RELEASE_ARTIFACTS <= _iter_upload_artifacts(jobs)
    assert "./scripts/provision-aca-private-origin.sh" in all_scripts
    assert "./scripts/smoke-release-gates.sh" in all_scripts
    assert "./scripts/validate-aca-phase3-auth.sh" not in all_scripts
    build_artifacts = {
        step.get("with", {}).get("name")
        for step in jobs["build_images"].get("steps", [])
        if step.get("uses") == "actions/upload-artifact@v4"
    }
    observe_artifacts = {
        step.get("with", {}).get("name")
        for step in jobs["observe_production_canary"].get("steps", [])
        if step.get("uses") == "actions/upload-artifact@v4"
    }
    assert "image-metadata" in build_artifacts
    assert "production-rollout-state" in observe_artifacts



def test_release_workflow_preserves_rollout_modes_approvals_and_oidc_login():
    workflow = _load_yaml(RELEASE_WORKFLOW_PATH)
    jobs = workflow["jobs"]
    deploy_staging_scripts = "\n".join(_iter_run_steps(jobs["deploy_staging"]))
    deploy_production_scripts = "\n".join(_iter_run_steps(jobs["deploy_production_canary"]))
    rollback_scripts = "\n".join(_iter_run_steps(jobs["rollback_production_canary"]))
    promote_scripts = "\n".join(_iter_run_steps(jobs["promote_production_full"]))
    deploy_production_env = next(
        step["env"]
        for step in jobs["deploy_production_canary"]["steps"]
        if step.get("name") == "Run production deployment script"
    )
    all_uses = [
        step.get("uses")
        for job in jobs.values()
        for step in job.get("steps", [])
        if isinstance(step.get("uses"), str)
    ]

    assert "ROLLOUT_MODE=reconcile" in deploy_staging_scripts
    assert "ROLLOUT_MODE=canary" in deploy_production_scripts
    assert deploy_production_env["CANARY_TRAFFIC_PERCENT"] == "${{ vars.CANARY_TRAFFIC_PERCENT }}"
    assert deploy_production_env["STABLE_TRAFFIC_PERCENT"] == "${{ vars.STABLE_TRAFFIC_PERCENT }}"
    assert "ROLLOUT_MODE=rollback" in rollback_scripts
    assert "ROLLOUT_MODE=promote" in promote_scripts
    assert jobs["await_prod_approval"]["environment"] == "production-canary-approval"
    assert jobs["await_full_promotion"]["environment"] == "production-full-approval"
    assert jobs["rollback_production_canary"]["if"] == "${{ !inputs.staging_only && failure() }}"
    assert jobs["observe_production_canary"]["if"] == "${{ !inputs.staging_only && success() }}"
    assert jobs["await_prod_approval"]["if"] == "${{ !inputs.staging_only }}"
    assert jobs["await_full_promotion"]["if"] == "${{ !inputs.staging_only }}"
    assert "azure/login@v2" in all_uses
    assert workflow["permissions"]["id-token"] == "write"
