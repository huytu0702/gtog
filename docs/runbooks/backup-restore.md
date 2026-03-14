# Backup and Restore Runbook

## Purpose

Define how to recover critical metadata, uploaded content, and rebuildable search/index state for the GToG production deployment.

## Scope

This runbook covers:
- Cosmos DB metadata, job state, and lease state
- Blob Storage uploaded content and generated artifacts
- Azure AI Search rebuild procedure
- Restore validation requirements
- RPO and RTO targets
- Drill cadence

## Data Classification

### Source of truth
The following are treated as authoritative system state:
- Cosmos DB metadata
- Cosmos DB job records and leases
- Blob Storage uploaded files
- Blob Storage generated or retained artifacts where configured

### Rebuildable state
The following may be rebuilt after recovery:
- Azure AI Search indexes
- derivative search artifacts that can be regenerated from Blob and Cosmos source data

## Recovery Objectives

### Target RPO
- Cosmos metadata and job state: low-loss target defined by platform backup capability
- Blob uploaded content: low-loss target defined by storage protection settings
- Azure AI Search: rebuildable; no strict point-in-time guarantee assumed on Free SKU

### Target RTO
- Metadata and app availability should be restored as quickly as practical using the most recent recoverable state
- Search serving may be temporarily degraded while indexes are rebuilt

Exact operational targets should be recorded in release or operations records for each environment.

## Cosmos Restore Procedure

### Restore triggers
Use Cosmos restore when:
- metadata is corrupted or deleted
- job state is lost or corrupted
- lease state prevents safe recovery and cannot be corrected manually

### Restore steps
1. Identify the affected containers and time window.
2. Confirm the restore source and restore point available in the current Cosmos backup policy.
3. Restore the affected data into a controlled recovery target.
4. Validate restored metadata before reintroducing it into the active service path.
5. Reconcile any in-flight jobs carefully to avoid duplicate final artifacts.

### Cosmos restore verification
After restore, verify:
- collections metadata is present and internally consistent
- job records are readable and sane
- lease state does not assign multiple active owners to a single job
- application startup and readiness checks succeed

## Blob Recovery Procedure

### Recovery triggers
Use Blob recovery when:
- uploaded files are deleted or overwritten
- retained artifacts are corrupted or missing

### Recovery steps
1. Identify the affected container, blob paths, and time window.
2. Use configured soft delete and versioning features where available.
3. Recover the required files into a controlled path.
4. Verify checksums, sizes, or expected document inventory where possible.
5. Reconnect recovered data to metadata only after integrity is confirmed.

### Blob recovery verification
After recovery, verify:
- expected files exist
- files are readable by the application
- metadata points to valid blob paths
- dependent workflows resume normally

## Azure AI Search Rebuild Procedure

### Rebuild triggers
Rebuild AI Search when:
- index contents are inconsistent with source data
- index is lost or unavailable
- a restore leaves search in an unknown state

### Rebuild assumptions
- AI Search is treated as rebuildable from Blob and Cosmos source-of-truth data
- Free SKU limitations may affect performance and operational flexibility

### Rebuild steps
1. Confirm source metadata and source files are healthy.
2. Identify the index or indexes that must be rebuilt.
3. Re-run the application-approved indexing or rebuild flow.
4. Verify index population before enabling dependent production traffic where necessary.

### AI Search rebuild verification
After rebuild, verify:
- readiness checks pass
- expected query paths work
- search results are returned for known validation cases
- no obvious mismatch remains between metadata and searchable content

## Restore Validation Checklist

After any restore or rebuild, validate:
- `app.<domain>` remains reachable if frontend is in scope
- `api.<domain>/health` and `/health/readiness` behave as expected
- authentication still works
- collection CRUD works if relevant to the incident
- indexing/job workflows resume safely
- direct-origin denial remains intact
- logs and alerts reflect the recovery event clearly

## Evidence to Retain

For every restore or drill, keep:
- incident or drill date and owner
- affected resources
- selected restore source or restore point
- command output or platform screenshots
- validation results
- unresolved caveats or follow-up items

## Drill Cadence

Run recovery drills on a regular schedule.

Minimum recommendation:
- validate at least one restore scenario in staging on a recurring basis
- re-check AI Search rebuild procedure after major indexing or schema changes
- refresh evidence after major platform or pipeline changes

## Notes on AI Search Free SKU

The current production design accepts Azure AI Search Free SKU as a temporary exception.

Implications:
- full private-network alignment is not available for Search
- restore posture for Search depends more heavily on rebuildability than isolation guarantees
- this exception must remain documented in release records until upgraded
