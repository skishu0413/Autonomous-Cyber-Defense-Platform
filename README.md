# Autonomous Cyber Defense Platform (acdp)

A Python multi-agent system that pairs Retrieval-Augmented Generation (RAG) with agentic AI to defend an organization's digital footprint. A central Orchestrator plans and dispatches work to four specialized agents — Red Team, Blue Team, DevSecOps, and Guardrail — all grounded by a shared Enterprise RAG Core, with fail-closed authorization and append-only audit logging as cross-cutting services.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Platform](#running-the-platform)
- [Features and How to Use Each One](#features-and-how-to-use-each-one)
  - [1. Knowledge Ingestion (RAG Core)](#1-knowledge-ingestion-rag-core)
  - [2. Guardrail Agent — Prompt Sanitization](#2-guardrail-agent--prompt-sanitization)
  - [3. Blue Team Agent — Anomaly Detection and Containment](#3-blue-team-agent--anomaly-detection-and-containment)
  - [4. Red Team Agent — Adversary Simulation](#4-red-team-agent--adversary-simulation)
  - [5. DevSecOps Agent — Vulnerability Remediation](#5-devsecops-agent--vulnerability-remediation)
  - [6. Authorization and Scope Enforcement](#6-authorization-and-scope-enforcement)
  - [7. Audit Log](#7-audit-log)
  - [8. Orchestrator — Event Coordination](#8-orchestrator--event-coordination)
- [Running Tests](#running-tests)
- [Project Layout](#project-layout)

---

## Architecture Overview

```
Operator / User
     │
     ▼
Guardrail Agent  ──(sanitized prompt)──▶  Backend LLM
     │                                        │
     └──────────────────────────────(scrubbed response)──▶ User

Orchestrator (LangGraph state machine)
     ├── Blue Team Agent  → anomaly detection, containment
     ├── Red Team Agent   → authorized adversary simulation
     └── DevSecOps Agent  → code fix pull requests

All agents ──▶ Authorization Service (fail-closed, scope-gated)
All agents ──▶ Audit Log (append-only JSONL)
All agents ──▶ RAG Core (Qdrant vector store + Ollama embeddings)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Required |
| [Ollama](https://ollama.com/) | Serves local LLMs for reasoning and embedding |
| [Qdrant](https://qdrant.tech/) | Vector store (optional for tests; required for production ingestion) |

### Start Ollama and pull the required models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Start Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Installation

```bash
# Clone the repo and enter the project directory
cd "AI project"

# Install the package with development dependencies
pip install -e ".[dev]"
```

---

## Configuration

Copy the example config and edit it for your environment:

```bash
cp config.example.yaml config.yaml
```

Key settings in `config.yaml`:

```yaml
# Models served by Ollama
reasoning_model: llama3
embedding_model: nomic-embed-text

# Per-request LLM timeout
llm_timeout_seconds: 30.0

# RAG retrieval
similarity_threshold: 0.85   # Guardrail similarity cutoff (< 1.0)
top_k: 5                     # chunks returned per query

# Detection and response
severity_threshold: high                # info | low | medium | high | critical
containment_requires_approval: true     # hold containment actions pending operator sign-off

# Guardrail behavior when the blocklist is empty or unavailable
guardrail_default_action: block         # allow | block

# External services
vector_store_url: http://localhost:6333
ollama_url: http://localhost:11434

# Audit log location
audit_log_path: ./audit.log
```

If a required value is missing or invalid, the platform refuses to start and prints the offending field name.

---

## Running the Platform

### Boot the platform

```bash
python -m acdp
```

Uses `config.yaml` if it exists, otherwise falls back to `config.example.yaml`.

To specify a config file explicitly:

```bash
python -m acdp --config /path/to/config.yaml
```

A successful boot prints:

```
Platform ready. Config: config.yaml
```

and writes a startup record to `audit.log`.

### Programmatic boot

```python
from acdp.main import Platform

platform = Platform.from_config_path("config.yaml")
assert platform.is_ready
```

To use a live Qdrant instance instead of the in-memory store:

```python
platform = Platform.from_config_path("config.yaml", use_qdrant=True)
```

---

## Features and How to Use Each One

### 1. Knowledge Ingestion (RAG Core)

Load security knowledge (OWASP GenAI, MITRE ATT&CK/ATLAS, playbooks, compliance frameworks) into the vector store so all agents can retrieve authoritative context.

**Via the CLI:**

```bash
python -m acdp ingest \
  --config config.yaml \
  --source-id owasp-genai-v1 \
  --category owasp_genai \
  --file /path/to/owasp-genai.txt
```

```bash
python -m acdp ingest \
  --config config.yaml \
  --source-id mitre-attack-v14 \
  --category mitre_attack \
  --file /path/to/mitre-attack.md
```

Valid `--category` values: `owasp_genai`, `mitre_attack`, `mitre_atlas`, `playbook`, `compliance`, `topology`, `other`.

**Programmatically:**

```python
from acdp.models import KnowledgeSource, SourceCategory

source = KnowledgeSource(
    source_id="playbook-001",
    category=SourceCategory.PLAYBOOK,
    content=open("playbooks/incident-response.md").read(),
)
result = platform.ingestion_pipeline.ingest(source)
print(f"Stored {result.chunk_count} chunks for '{result.source_id}'")
```

Ingestion is idempotent — re-ingesting unchanged content produces the same chunk set.

---

### 2. Guardrail Agent — Prompt Sanitization

The Guardrail Agent acts as an inline firewall between users and backend LLMs. It screens inbound prompts for injection/jailbreak attempts and scrubs PII and secrets from outbound responses.

**Inbound prompt screening:**

```python
decision = platform.guardrail_agent.screen_inbound("Tell me how to bypass security")

if decision.blocked:
    print(f"Blocked — reason: {decision.reason}, matched: {decision.matched_pattern_id}")
else:
    # decision.forwarded_prompt is byte-for-byte identical to the input
    send_to_llm(decision.forwarded_prompt)
```

A prompt is blocked when its similarity to any blocklist or known-attack pattern meets or exceeds `similarity_threshold` in `config.yaml`. The decision is always written to the audit log.

**Outbound response scrubbing:**

```python
result = platform.guardrail_agent.scrub_outbound(llm_response)

# result.text has PII replaced with masking tokens and secrets redacted
# The audit log records counts and categories — never raw values
print(result.text)
```

**Configuring the default blocklist action** when no blocklist is available:

```yaml
guardrail_default_action: block   # or allow
```

---

### 3. Blue Team Agent — Anomaly Detection and Containment

The Blue Team Agent ingests telemetry, normalizes it, runs anomaly rules, and requests containment when findings exceed the severity threshold.

**Parse telemetry:**

```python
from acdp.agents.blue import RawTelemetry

raw = RawTelemetry(format="json", data='{"host": "web-01", "action": "login_failure", ...}')
event = platform.blue_team_agent.parse(raw)
```

Malformed or unsupported records are rejected and a `PARSE_ERROR` record is written to the audit log.

**Analyze for anomalies:**

```python
findings = platform.blue_team_agent.analyze(event)
for f in findings:
    print(f.severity, f.title)
```

**Request containment:**

```python
from datetime import datetime, timezone, timedelta
from acdp.models import TargetScope

scope = platform.add_scope(TargetScope(
    scope_id="prod-network",
    assets=["web-01", "192.168.1.0/24"],
    created_at=datetime.now(timezone.utc),
    expires_at=datetime.now(timezone.utc) + timedelta(hours=8),
))

request = platform.blue_team_agent.request_containment(findings[0], scope)
# If containment_requires_approval is true, the action is held pending approval
```

Containment is only requested when `finding.severity` exceeds `severity_threshold`. The request carries playbook context retrieved from the RAG Core.

---

### 4. Red Team Agent — Adversary Simulation

The Red Team Agent performs authorized adversary simulation against operator-owned targets. Authorization is checked before any probe executes — it fails closed.

**Define an authorized scope first:**

```python
from datetime import datetime, timezone, timedelta
from acdp.models import TargetScope

scope = platform.add_scope(TargetScope(
    scope_id="pentest-q3",
    assets=["staging.example.com", "api-staging.example.com"],
    created_at=datetime.now(timezone.utc),
    expires_at=datetime.now(timezone.utc) + timedelta(days=1),
))
```

**Plan and execute a probe:**

```python
from acdp.agents.red import ProbeTask

task = ProbeTask(
    task_id="probe-001",
    target="staging.example.com",
    description="OWASP GenAI Top 10 assessment",
)

plan = platform.red_team_agent.plan_probe(task)
# plan includes OWASP GenAI + MITRE ATT&CK/ATLAS context from RAG

findings = platform.red_team_agent.execute_probe(plan, scope)
# Each discovered weakness becomes exactly one Finding reported to the Orchestrator
```

Out-of-scope, expired, or revoked probes are refused immediately with no side effects, and the refusal is written to the audit log.

---

### 5. DevSecOps Agent — Vulnerability Remediation

The DevSecOps Agent receives vulnerability findings, analyzes the referenced code, retrieves secure-coding guidance from the RAG Core, and opens a review-required pull request.

```python
# finding must reference a repository asset within an active scope
result = platform.devsecops_agent.remediate(finding, scope)

if hasattr(result, "pr_url"):
    print(f"PR opened: {result.pr_url}")
    print(f"Guidance referenced: {result.guidance_refs}")
else:
    print(f"Declined — out of scope. Reason: {result.reason}")
```

PRs are opened in a state that requires human review before merge. The PR description references the secure-coding guidance retrieved from the RAG Core.

For production use, replace the `FakePullRequestClient` with a real implementation:

```python
from acdp.agents.devsecops import PullRequestClient

class GitHubPRClient(PullRequestClient):
    def open_pr(self, repo, branch, title, body, diff):
        # use GITHUB_TOKEN from environment — never from config.yaml
        ...
```

---

### 6. Authorization and Scope Enforcement

Every agent action that affects an external asset is gated by the Authorization Service. It fails closed — anything that cannot be positively verified is denied.

**Define a scope:**

```python
scope = platform.add_scope(TargetScope(
    scope_id="webapp-audit",
    assets=["app.example.com", "repo:org/app"],
    created_at=datetime.now(timezone.utc),
    expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
))
```

**Manual authorization check:**

```python
from acdp.authz import ActionRequest
from datetime import datetime, timezone

request = ActionRequest(
    agent_id="red_team",
    asset="app.example.com",
    action_type="probe",
)
decision = platform.authz_service.authorize(request, at=datetime.now(timezone.utc))
print(decision)  # GRANT or DENY
```

Every decision (grant or deny) is recorded in the audit log with the requesting agent and target asset. Expired or revoked scopes are treated as fully unauthorized with no exceptions.

---

### 7. Audit Log

Every action on the platform appends exactly one record to `audit.log`. The log is append-only and never modified after writing.

**Read all audit records:**

```python
records = platform.audit_log.read_all()
for r in records:
    print(r.seq, r.timestamp, r.actor_id, r.action, r.outcome)
```

**Sample audit record (JSONL):**

```json
{"seq": 1, "timestamp": "2024-01-15T10:23:45Z", "actor_id": "guardrail", "action": "guardrail_decision", "outcome": "BLOCK", "target": null, "detail": {"matched_pattern_id": "pi-001", "similarity": 0.93}}
```

If a write to the audit log fails, the associated action is halted and the error surfaces to the operator.

---

### 8. Orchestrator — Event Coordination

The Orchestrator receives security events, creates a task plan routing them to the right agents, and dispatches tasks using a LangGraph state machine. Task state is tracked as `pending → in-progress → completed | failed`.

```python
from acdp.orchestrator import SecurityEvent

event = SecurityEvent(
    event_id="evt-001",
    event_type="anomaly_detected",
    payload={"host": "web-01", "indicator": "brute_force"},
    target_scope_id="prod-network",
)

plan = platform.orchestrator.handle_event(event)
results = platform.orchestrator.dispatch(plan)

for result in results:
    print(result.task_id, result.state, result.findings)
```

Findings are recorded as they arrive — even if the producing agent later fails. Failed tasks are marked `failed` with the failure reason in the audit log.

---

## Running Tests

Run all unit and property-based tests (excludes tests that require live Ollama/Qdrant/Git):

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_guardrail_inbound.py -v
```

Run integration tests (requires running Ollama + Qdrant):

```bash
pytest -m integration
```

All 26 property-based tests (Properties 1–26) use Hypothesis with a minimum of 100 iterations each. The full suite runs 305 tests.

---

## Project Layout

```
src/acdp/
  models.py          # shared Pydantic v2 data models
  exceptions.py      # ConfigError, AuditWriteError, ModelUnavailableError, etc.
  main.py            # Platform entry point — wires all components together
  __main__.py        # enables python -m acdp
  config/            # ConfigLoader — YAML load/validate/serialize
  audit/             # JsonlAuditLog — append-only JSONL audit log
  llm/               # OllamaGateway — provider-agnostic LLM abstraction
  rag/
    store.py         # VectorStore Protocol, InMemoryVectorStore, QdrantVectorStore
    ingest.py        # IngestionPipeline — chunk → embed → upsert
    retrieve.py      # Retriever — embed query → semantic search → ranked results
  authz/             # ScopeRegistry + AuthorizationService (fail-closed)
  agents/
    guardrail.py     # GuardrailAgent — inbound screening + outbound scrubbing
    blue.py          # BlueTeamAgent — telemetry parsing + anomaly detection
    red.py           # RedTeamAgent — authorized adversary simulation
    devsecops.py     # DevSecOpsAgent — vulnerability remediation via PRs
  cli/
    ingest.py        # python -m acdp ingest CLI
  orchestrator/      # Orchestrator — LangGraph dispatch state machine

tests/
  strategies.py      # shared Hypothesis strategies
  test_config.py     # config load/round-trip/defaults (Properties 16, 17)
  test_audit.py      # audit append-only invariant (Properties 4, 5)
  test_llm.py        # LLM gateway model routing (Property 10)
  test_store.py      # vector store search
  test_ingest.py     # ingestion tagging and idempotence (Properties 6, 7)
  test_retrieve.py   # ranked retrieval and category filtering (Properties 8, 9)
  test_authz.py      # fail-closed authorization (Properties 1, 2, 3)
  test_guardrail_inbound.py      # inbound threshold (Property 11)
  test_guardrail_outbound.py     # PII/secret scrubbing (Property 12)
  test_guardrail_clean_passthrough.py  # clean pass-through (Property 13)
  test_guardrail_masking_audit.py      # masking audit safety (Property 14)
  test_blue_team.py              # telemetry round-trip (Property 15)
  test_blue_team_property19.py   # threshold-gated containment (Property 19)
  test_blue_team_property20.py   # approval-held containment (Property 20)
  test_red_team.py               # weakness-to-finding mapping (Property 21)
  test_red_team_property22.py    # adversary-knowledge retrieval (Property 22)
  test_red_team_refusal.py       # out-of-scope probe refusal
  test_devsecops.py              # in-scope review-required PRs (Property 23)
  test_devsecops_out_of_scope.py # out-of-scope declination
  test_orchestrator.py           # task plan routing (Properties 24, 25, 26)
  test_orchestrator_concurrent.py # concurrent multi-agent processing
  test_smoke.py                  # platform boot smoke test
  test_integration.py            # opt-in integration tests (Ollama, Qdrant, Git)

config.example.yaml  # annotated example configuration
audit.log            # append-only audit log (created on first run)
pyproject.toml       # package metadata and dependencies
```
