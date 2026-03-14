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
    "deploy_production",
    "smoke_production",
]
EXPECTED_RELEASE_ARTIFACTS = {
    "deploy-production-log",
    "deploy-staging-log",
    "image-metadata",
    "phase3-auth-origin-validation",
    "phase3-auth-origin-validation-production",
    "smoke-production-report",
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



def _step_named(job: dict, step_name: str) -> dict:
    return next(step for step in job.get("steps", []) if step.get("name") == step_name)



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
    assert jobs["deploy_production"]["needs"] == ["await_prod_approval"]
    assert jobs["smoke_production"]["needs"] == ["deploy_production"]



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
    production_artifacts = {
        step.get("with", {}).get("name")
        for step in jobs["deploy_production"].get("steps", [])
        if step.get("uses") == "actions/upload-artifact@v4"
    }
    assert "image-metadata" in build_artifacts
    assert "deploy-production-log" in production_artifacts



def test_release_workflow_builds_environment_specific_frontend_images_and_reuses_outputs():
    workflow = _load_yaml(RELEASE_WORKFLOW_PATH)
    jobs = workflow["jobs"]
    build_scripts = "\n".join(_iter_run_steps(jobs["build_images"]))
    build_outputs = jobs["build_images"]["outputs"]
    deploy_staging_env = _step_named(jobs["deploy_staging"], "Run staging deployment script")["env"]
    deploy_production_env = _step_named(jobs["deploy_production"], "Run production deployment script")["env"]

    assert build_outputs["backend_image"] == "${{ steps.metadata.outputs.backend_image }}"
    assert build_outputs["worker_image"] == "${{ steps.metadata.outputs.worker_image }}"
    assert build_outputs["frontend_image_staging"] == "${{ steps.metadata.outputs.frontend_image_staging }}"
    assert build_outputs["frontend_image_production"] == "${{ steps.metadata.outputs.frontend_image_production }}"
    assert "NEXT_PUBLIC_API_BASE_URL_STAGING" in build_scripts
    assert "NEXT_PUBLIC_API_BASE_URL_PRODUCTION" in build_scripts
    assert "frontend_image_staging" in build_scripts
    assert "frontend_image_production" in build_scripts
    assert deploy_staging_env["FRONTEND_IMAGE"] == "${{ needs.build_images.outputs.frontend_image_staging }}"
    assert deploy_production_env["FRONTEND_IMAGE"] == "${{ needs.build_images.outputs.frontend_image_production }}"
    assert deploy_staging_env["API_IMAGE"] == "${{ needs.build_images.outputs.backend_image }}"
    assert deploy_production_env["WORKER_IMAGE"] == "${{ needs.build_images.outputs.worker_image }}"



def test_release_workflow_preserves_deploy_contract_approvals_and_oidc_login():
    workflow = _load_yaml(RELEASE_WORKFLOW_PATH)
    jobs = workflow["jobs"]
    deploy_staging_scripts = "\n".join(_iter_run_steps(jobs["deploy_staging"]))
    deploy_production_scripts = "\n".join(_iter_run_steps(jobs["deploy_production"]))
    deploy_staging_env = _step_named(jobs["deploy_staging"], "Run staging deployment script")["env"]
    deploy_production_env = _step_named(jobs["deploy_production"], "Run production deployment script")["env"]
    smoke_staging_env = _step_named(jobs["smoke_staging"], "Run staging smoke gates")["env"]
    smoke_production_env = _step_named(jobs["smoke_production"], "Run production smoke gates")["env"]
    all_uses = [
        step.get("uses")
        for job in jobs.values()
        for step in job.get("steps", [])
        if isinstance(step.get("uses"), str)
    ]

    assert "ROLLOUT_MODE=reconcile" in deploy_staging_scripts
    assert "ROLLOUT_MODE=reconcile" in deploy_production_scripts
    assert deploy_staging_env["CREATE_APPS"] == "true"
    assert deploy_production_env["CREATE_APPS"] == "true"
    assert deploy_staging_env["CONFIGURE_EASY_AUTH"] == "true"
    assert deploy_production_env["CONFIGURE_EASY_AUTH"] == "true"
    assert deploy_staging_env["FRONTEND_APP_NAME"] == "${{ vars.FRONTEND_APP_NAME }}"
    assert deploy_staging_env["APP_PUBLIC_HOSTNAME"] == "${{ vars.APP_PUBLIC_HOSTNAME }}"
    assert deploy_staging_env["API_PUBLIC_HOSTNAME"] == "${{ vars.API_PUBLIC_HOSTNAME }}"
    assert deploy_production_env["FRONTEND_APP_NAME"] == "${{ vars.FRONTEND_APP_NAME }}"
    assert deploy_production_env["APP_PUBLIC_HOSTNAME"] == "${{ vars.APP_PUBLIC_HOSTNAME }}"
    assert deploy_production_env["API_PUBLIC_HOSTNAME"] == "${{ vars.API_PUBLIC_HOSTNAME }}"
    assert smoke_staging_env["APP_BASE_URL"] == "https://${{ vars.APP_PUBLIC_HOSTNAME }}"
    assert smoke_staging_env["API_BASE_URL"] == "https://${{ vars.API_PUBLIC_HOSTNAME }}"
    assert smoke_production_env["APP_BASE_URL"] == "https://${{ vars.APP_PUBLIC_HOSTNAME }}"
    assert smoke_production_env["API_BASE_URL"] == "https://${{ vars.API_PUBLIC_HOSTNAME }}"
    assert jobs["await_prod_approval"]["environment"] == "production-approval"
    assert jobs["await_prod_approval"]["if"] == "${{ !inputs.staging_only }}"
    assert jobs["deploy_production"]["environment"] == "production"
    assert jobs["smoke_production"]["environment"] == "production"
    assert "azure/login@v2" in all_uses
    assert workflow["permissions"]["id-token"] == "write"
