# ACDP — Usage Guide

> **Autonomous Cyber Defense Platform** — a RAG-backed multi-agent security platform  
> Version `0.1.0` · Python 3.11+

---

## Table of Contents

1. [How It Works — Big Picture](#1-how-it-works--big-picture)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Running the Platform](#5-running-the-platform)
6. [Ingesting Knowledge Sources](#6-ingesting-knowledge-sources)
7. [The Four Agents](#7-the-four-agents)
   - [Guardrail Agent](#guardrail-agent)
   - [Blue Team Agent](#blue-team-agent)
   - [Red Team Agent](#red-team-agent)
   - [DevSecOps Agent](#devsecops-agent)
8. [How Agents Work Together](#8-how-agents-work-together)
9. [Authorization & Scopes](#9-authorization--scopes)
10. [Audit Log](#10-audit-log)
11. [After Ingestion — How Agents Use the Knowledge Base](#13-after-ingestion--how-agents-use-the-knowledge-base)
12. [Running Tests](#11-running-tests)
13. [Troubleshooting](#12-troubleshooting)

---

## 1. How It Works — Big Picture

```
                      ┌─────────────────────────────────┐
                      │       Central Orchestrator       │
                      │  (plans tasks, dispatches work)  │
                      └──────────────┬──────────────────┘
                                     │
          ┌──────────────┬───────────┼────────────┬──────────────┐
          │              │           │            │              │
   ┌──────▼─────┐ ┌──────▼────┐ ┌───▼──────┐ ┌──▼───────────┐
   │  Guardrail │ │ Blue Team │ │ Red Team │ │  DevSecOps   │
   │   Agent    │ │   Agent   │ │  Agent   │ │    Agent     │
   │ (firewall) │ │ (defense) │ │(offense) │ │ (fix code)   │
   └──────┬─────┘ └──────┬────┘ └───┬──────┘ └──┬───────────┘
          └──────────────┴──────────┴────────────┘
                                     │
                      ┌──────────────▼──────────────────┐
                      │       Enterprise RAG Core        │
                      │  OWASP · MITRE · Playbooks ·     │
                      │  Compliance · Infrastructure     │
                      └─────────────────────────────────┘
```

The platform works in layers:

| Layer | Components | Role |
|---|---|---|
| **Foundation** | Config, Audit Log, LLM Gateway | Boot, logging, model routing |
| **Knowledge Base** | Vector Store, Ingestion, Retriever | Store and query security knowledge |
| **Policy** | Scope Registry, Authorization Service | Define who can act on what |
| **Agents** | Guardrail, Blue Team, Red Team, DevSecOps | Specialized security workers |
| **Orchestration** | Orchestrator (LangGraph) | Route events to agents, track task state |

Every agent action is gated by the **Authorization Service** (fail-closed) and every decision is written to the **Audit Log** before it takes effect.

---

## 2. Prerequisites

You need three external services running before starting ACDP:

### Ollama (local LLM server)

```bash
# Install: https://ollama.com
ollama serve

# Pull the required models
ollama pull llama3
ollama pull nomic-embed-text
```

Default URL: `http://localhost:11434`

### Qdrant (vector store)

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or via Homebrew
brew install qdrant
qdrant
```

Default URL: `http://localhost:6333`

### Python 3.11+

```bash
python --version   # must be 3.11 or higher
```

---

## 3. Installation

```bash
# Clone or open the project
cd "AI project"

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install the package and all dependencies
pip install -e .

# Install optional extras for file ingestion
pip install pypdf openpyxl       # PDF and Excel support
```

Verify the install:

```bash
python -m acdp --help
```

---

## 4. Configuration

Copy the example config and edit for your environment:

```bash
cp config.example.yaml config.yaml
```

Open `config.yaml` and review these key settings:

```yaml
# Models served by Ollama
reasoning_model: llama3             # for agent reasoning and code generation
embedding_model: nomic-embed-text   # for knowledge base embeddings

# How long to wait for an LLM response
llm_timeout_seconds: 30.0

# Guardrail similarity cutoff — prompts above this score are blocked
similarity_threshold: 0.85

# Number of knowledge chunks returned per query
top_k: 5

# Findings at or below this severity are NOT escalated to containment
severity_threshold: high            # info | low | medium | high | critical

# Require human approval before any containment action executes
containment_requires_approval: true

# What to do when the blocklist is empty or unavailable
guardrail_default_action: block     # allow | block

# External service URLs
vector_store_url: http://localhost:6333
ollama_url: http://localhost:11434

# Where the audit log is written
audit_log_path: ./audit.log
```

---

## 5. Running the Platform

### Boot the platform

```bash
python -m acdp
```

Or with an explicit config path:

```bash
python -m acdp --config /path/to/config.yaml
```

You will see an animated startup sequence showing each layer initialising, followed by a **● ACDP READY** panel confirming all four agents are live.

### Programmatic boot

```python
from acdp.main import Platform

platform = Platform.from_config_path("config.yaml")
assert platform.is_ready
```

---

## 6. Ingesting Knowledge Sources

Before agents can retrieve context, you need to load security knowledge into the vector store. ACDP supports **text**, **PDF**, and **Excel** files.

### Supported categories

| Category | `--category` value | What to ingest |
|---|---|---|
| OWASP GenAI Top 10 | `owasp_genai` | OWASP GenAI PDF/text |
| MITRE ATT&CK | `mitre_attack` | ATT&CK matrix (Excel or JSON) |
| MITRE ATLAS | `mitre_atlas` | ATLAS matrix |
| Security Playbooks | `playbook` | Internal runbooks / response guides |
| Compliance | `compliance` | ISO 27001, SOC 2, NIST frameworks |
| Infrastructure | `topology` | Network topology, asset inventory |
| Other | `other` | Any other reference material |

### Ingest a PDF

```bash
python -m acdp ingest \
  --config config.yaml \
  --source-id owasp-genai-v1 \
  --category owasp_genai \
  --file /path/to/owasp-genai.pdf
```

### Ingest an Excel file

```bash
python -m acdp ingest \
  --config config.yaml \
  --source-id mitre-attack-v14 \
  --category mitre_attack \
  --file /path/to/enterprise-attack.xlsx
```

### Ingest a text or Markdown file

```bash
python -m acdp ingest \
  --config config.yaml \
  --source-id incident-playbook-001 \
  --category playbook \
  --file /path/to/playbook.md
```

Each ingestion command:
1. Reads and parses the file
2. Splits it into overlapping chunks
3. Embeds each chunk via the Ollama embedding model
4. Stores the embedded chunks in Qdrant with the `source_id` and `category` tags

---

## 7. The Four Agents

---

### Guardrail Agent

**Purpose:** Inline firewall between users and backend LLMs.

**Inbound screening** — checks every user prompt before it reaches an LLM:

```python
from acdp.agents.guardrail_agent import GuardrailAgent

decision = platform.guardrail_agent.screen_inbound(prompt)

if decision.action == "block":
    print(f"Blocked: {decision.reason}")
    print(f"Matched pattern: {decision.matched_pattern_id}")
else:
    # Safe to forward to the LLM
    print("Forwarded")
```

A prompt is **blocked** if its similarity to any known-attack pattern in the blocklist meets or exceeds `similarity_threshold`. Otherwise it is forwarded byte-for-byte unchanged.

**Outbound scrubbing** — strips PII and secrets from LLM responses before they reach the user:

```python
result = platform.guardrail_agent.scrub_outbound(llm_response)

print(result.response)        # scrubbed text
print(result.masked)          # True if anything was replaced
for s in result.summaries:
    print(s.category, s.count)  # e.g. "email 2", "api_key 1"
```

Detected and masked automatically:

| Type | Example | Replaced with |
|---|---|---|
| Email address | `user@example.com` | `[MASKED_EMAIL]` |
| US SSN | `123-45-6789` | `[MASKED_SSN]` |
| Credit card | `4111 1111 1111 1111` | `[MASKED_CREDIT_CARD]` |
| Financial account | `00123456789` | `[MASKED_FINANCIAL_ACCOUNT]` |
| API key / secret | `sk-live-abc123...` | `[REDACTED_API_KEY]` |

---

### Blue Team Agent

**Purpose:** Parse telemetry, detect anomalies, request containment.

**Step 1 — Parse raw telemetry** (supports JSON, Syslog RFC 3164/5424, CEF):

```python
from acdp.agents.blue_team_agent import BlueTeamAgent

event = platform.blue_team_agent.parse(raw_log_line)
print(event.host, event.actor, event.severity)
```

**Step 2 — Detect anomalies:**

```python
findings = platform.blue_team_agent.analyze(event)

for finding in findings:
    print(finding.severity.value, finding.title)
    print(finding.detail)
```

Built-in detection rules:

| Rule | Trigger | Severity |
|---|---|---|
| Repeated failed logins | 5+ failures from same actor | MEDIUM / HIGH |
| High-severity event | Event already marked HIGH or CRITICAL | HIGH / CRITICAL |
| Suspicious actor | `root`, `admin`, `anonymous`, `guest`, `system` | MEDIUM |
| Port scanning | Keywords: `nmap`, `masscan`, `scan detected` | HIGH |
| Privilege escalation | Keywords: `sudo`, `setuid`, `chmod 777` | CRITICAL |

**Step 3 — Request containment** (only for findings above `severity_threshold`):

```python
from acdp.models import TargetScope
from datetime import datetime, timezone, timedelta

scope = platform.add_scope(TargetScope(
    scope_id="scope-infra-01",
    assets=["web-01", "db-01"],
    created_at=datetime.now(timezone.utc),
    expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
))

request = platform.blue_team_agent.request_containment(finding, scope)
print(request.status)   # "approved" | "pending_approval" | "denied"
```

When `containment_requires_approval: true`, requests are held as `"pending_approval"` instead of being auto-executed.

---

### Red Team Agent

**Purpose:** Authorized adversary simulation — probe your own systems before attackers do.

**Step 1 — Plan a probe** (retrieves OWASP GenAI + MITRE ATT&CK + MITRE ATLAS context):

```python
from acdp.agents.red_team_agent import ProbeTask

task = ProbeTask(
    task_id="task-001",
    event_id="evt-001",
    target_asset="api.example.com",
    description="Test for prompt injection and model extraction vulnerabilities",
)

plan = platform.red_team_agent.plan_probe(task)
print(plan.weaknesses)       # derived from the description
print(plan.context_refs)     # source IDs of retrieved adversary knowledge
```

**Step 2 — Execute the probe** (authorization is checked first, fail-closed):

```python
findings = platform.red_team_agent.execute_probe(plan, scope)

for finding in findings:
    print(finding.title)      # one Finding per weakness in the plan
    print(finding.detail)
    print(finding.context_refs)  # linked back to OWASP/MITRE sources
```

If the target is **out of scope**, expired, or revoked, the probe is refused entirely — no findings, no side effects — and the refusal is written to the audit log.

> **Important:** The Red Team Agent only probes assets you have explicitly authorized via a `TargetScope`. Never configure it to target external systems you do not own.

---

### DevSecOps Agent

**Purpose:** Automatically remediate vulnerabilities in code repositories via pull requests.

**Remediate a finding:**

```python
from acdp.agents.devsecops_agent import DevSecOpsAgent

result = platform.devsecops_agent.remediate(finding, scope)
```

If the finding's asset (repository) is **in scope**:
1. Retrieves secure-coding guidance from the RAG Core
2. Sends the finding + guidance to the LLM to generate a fix patch
3. Opens a pull request on the repository — always with `requires_review=True`
4. Records a `PR_OPENED` audit entry with the PR ID and guidance source references

```python
from acdp.agents.devsecops_agent import PullRequest, Declination

if isinstance(result, PullRequest):
    print(f"PR opened: {result.pr_id}")
    print(f"Branch: fix/vuln-{finding.finding_id[:8]}")
    print(result.body)
elif isinstance(result, Declination):
    print(f"Declined: {result.reason}")
```

If the repository is **out of scope**, a `Declination` is returned and the refusal is audited. No PR is opened.

**Providing a real Git client:**

```python
from acdp.agents.devsecops_agent import PullRequestClient, PullRequest
import os

class GitHubPRClient:
    def create_pr(self, repo, title, body, branch, patch) -> PullRequest:
        token = os.environ["GIT_TOKEN"]   # credentials MUST come from env vars
        # ... call GitHub API ...
```

---

## 8. How Agents Work Together

The **Orchestrator** ties everything together. It receives `SecurityEvent` objects, creates a `TaskPlan` routing each event to the right agent(s), and dispatches the tasks through a LangGraph state machine.

### Event routing

| Event type | Routed to |
|---|---|
| `telemetry` | Blue Team + Guardrail |
| `probe_request` | Red Team + Guardrail |
| `vulnerability_finding` | DevSecOps + Guardrail |
| `prompt` | Guardrail |
| `unknown` | Guardrail |

Guardrail is always included as a safety layer regardless of event type.

### Full end-to-end example

```python
import uuid
from datetime import datetime, timezone, timedelta
from acdp.main import Platform
from acdp.models import SecurityEvent, SecurityEventType, TargetScope

# 1. Boot the platform
platform = Platform.from_config_path("config.yaml")

# 2. Define an authorized scope
scope = platform.add_scope(TargetScope(
    scope_id="scope-prod-01",
    assets=["web-01", "db-01", "api.example.com"],
    created_at=datetime.now(timezone.utc),
    expires_at=datetime.now(timezone.utc) + timedelta(hours=8),
))

# 3. Send a telemetry event (e.g. from a SIEM alert)
event = SecurityEvent(
    event_id=str(uuid.uuid4()),
    event_type=SecurityEventType.TELEMETRY,
    target_scope_id=scope.scope_id,
    payload={"raw": "Jan 15 03:42:11 web-01 sshd[1234]: Failed password for root"},
    timestamp=datetime.now(timezone.utc),
)

# 4. Orchestrator plans and dispatches tasks
plan    = platform.orchestrator.handle_event(event)
results = platform.orchestrator.dispatch(plan)

# 5. Review results
for result in results:
    print(f"Agent: {result.task.assigned_agent}  State: {result.task.state.value}")
    for finding in result.findings:
        print(f"  [{finding.severity.value.upper()}] {finding.title}")
```

### Task lifecycle

```
PENDING  →  IN_PROGRESS  →  COMPLETED
                         ↘  FAILED  (failure_reason recorded in audit log)
```

Findings produced by an agent are always recorded in the audit log **before** any failure for that task is handled — so partial results are never lost.

---

## 9. Authorization & Scopes

Every destructive action (containment, probe, pull request) is gated by the **Authorization Service**. It grants a request **only** when the target asset is inside a currently active scope. Everything else is denied.

```python
# Define a scope
scope = platform.add_scope(TargetScope(
    scope_id="scope-vuln-scan",
    assets=["staging.example.com"],
    created_at=datetime.now(timezone.utc),
    expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
))

# Revoke a scope early
scope.revoked = True
```

A scope is **active** when:
- `revoked` is `False`
- the current time is before `expires_at`

All scope definitions and every authorization decision (grant or deny) are recorded in the audit log with the requesting agent ID and target asset.

---

## 10. Audit Log

Every platform action appends exactly one record to the audit log at `audit_log_path` (default: `./audit.log`). Records are written in JSONL format — one JSON object per line — and flushed to disk with `fsync` before the action proceeds.

**Reading the audit log:**

```python
records = platform.audit_log.read_all()

for record in records:
    print(record.seq, record.action.value, record.actor_id, record.outcome)
```

**Sample record:**

```json
{
  "seq": 12,
  "timestamp": "2024-01-15T03:42:12.001Z",
  "actor_id": "blue_team",
  "action": "authz_grant",
  "outcome": "grant",
  "target": "web-01",
  "detail": {
    "agent_id": "blue_team",
    "asset": "web-01",
    "action": "containment",
    "decision": "grant",
    "scope_id": "scope-prod-01"
  }
}
```

**Audit action types:**

| Action | When it's written |
|---|---|
| `finding_recorded` | A finding is produced by any agent |
| `task_failed` | An orchestrated task fails |
| `scope_defined` | An operator defines a new TargetScope |
| `authz_grant` | Authorization service grants a request |
| `authz_deny` | Authorization service denies a request |
| `guardrail_decision` | Guardrail blocks or forwards a prompt |
| `masking_applied` | Outbound scrubbing replaces PII/secrets |
| `parse_error` | Blue Team rejects a malformed telemetry record |
| `probe_refused` | Red Team refuses an out-of-scope probe |
| `pr_opened` | DevSecOps opens a pull request |
| `remediation_declined` | DevSecOps declines an out-of-scope finding |

---

## 13. After Ingestion — How Agents Use the Knowledge Base

This section explains exactly what happens inside each agent after you have run `python -m acdp ingest`. Understanding this flow helps you know why ingesting the right sources matters and what each agent is actually doing when it runs.

---

### What ingestion produces

When you ingest a file, the pipeline does three things:

```
Your file
   │
   ▼
Split into 512-character chunks (word-boundary aware)
   │
   ▼
Each chunk embedded → float vector  (via Ollama nomic-embed-text)
   │
   ▼
Stored in Qdrant with tags: source_id, category, ingested_at
```

Each chunk gets a deterministic `chunk_id` = `SHA256(source_id + index + text)`, so re-ingesting the same file updates rather than duplicates.

After ingestion, Qdrant holds something like:

```
chunk_id: a3f9...   source_id: owasp-genai-v1    category: owasp_genai    text: "LLM01: Prompt Injection..."    vector: [0.12, -0.43, ...]
chunk_id: b7d2...   source_id: owasp-genai-v1    category: owasp_genai    text: "Prompt injection attacks..."   vector: [0.09, -0.41, ...]
chunk_id: c1e8...   source_id: mitre-attack-v14  category: mitre_attack   text: "T1566 Phishing: Adversaries..." vector: [0.31,  0.11, ...]
chunk_id: d5f1...   source_id: incident-playbook  category: playbook       text: "Step 1: Isolate the host..."   vector: [-0.05, 0.62, ...]
```

---

### The retrieval mechanism (used by every agent)

Every agent that needs context calls `Retriever.retrieve(query, category)`:

```
Agent builds a query string
        │
        ▼
Retriever embeds the query   →   float vector  (same model as ingestion)
        │
        ▼
Vector Store computes cosine similarity against every stored chunk
        │
        ▼
Returns top-K chunks ranked by similarity score  (default K=5)
        │
        ▼
Agent uses chunk.text as grounding context
```

The key insight: **the query is embedded into the same vector space as the stored chunks**, so semantic similarity — not keyword matching — determines what gets returned.

---

### Guardrail Agent — what it retrieves and why

**When it runs:** On every inbound prompt, before the LLM sees it.

**What it queries:**

```python
# From guardrail_agent.py — screen_inbound()
for category in (OWASP_GENAI, MITRE_ATLAS, MITRE_ATTACK):
    chunks = retriever.retrieve(prompt, category=category)
    # Each chunk's source_id becomes a match candidate
    # Its score is compared against similarity_threshold
```

**How it uses the result:**

The Guardrail does NOT use the chunk text directly. It uses the **similarity score** as a signal:

```
prompt similarity to OWASP/MITRE chunk  ≥  similarity_threshold (0.85)
        │
        YES ─────────────────► BLOCK  (prompt is too similar to a known attack)
        │
        NO  ─────────────────► FORWARD (prompt is safe)
```

**What this means in practice:**

- Ingest `owasp-genai.pdf` → Guardrail can detect prompts that look like LLM01 Prompt Injection, LLM02 Insecure Output Handling, etc.
- Ingest `enterprise-attack.xlsx` → Guardrail can detect prompts that resemble MITRE ATT&CK techniques
- Without ingestion, the Guardrail falls back to its blocklist only (or the default action if the blocklist is empty)

---

### Blue Team Agent — what it retrieves and why

**When it runs:** After detecting a finding that exceeds `severity_threshold`, when requesting containment.

**What it queries:**

```python
# From blue_team_agent.py — request_containment()
query = f"containment playbook for {finding.title}: {finding.detail[:120]}"
chunks = retriever.retrieve(query, category=SourceCategory.PLAYBOOK)
```

**How it uses the result:**

```
Finding: "Repeated failed login attempts detected" on host web-01
        │
        ▼
Query: "containment playbook for Repeated failed login attempts detected: Actor 'root'..."
        │
        ▼
Retriever finds: "Step 1: Block source IP. Step 2: Force password reset..."   score: 0.91
                 "Isolate the host from the network using firewall rule..."    score: 0.87
        │
        ▼
chunk.source_id values → stored in ContainmentRequest.playbook_refs
```

The `playbook_refs` field on the returned `ContainmentRequest` tells operators which playbook sections grounded the containment decision. This is auditable provenance.

**What this means in practice:**

- Ingest your internal runbooks with `--category playbook` → Blue Team's containment requests are grounded in your actual procedures
- Without playbook ingestion, `playbook_refs` will be empty (containment still works but has no context)

---

### Red Team Agent — what it retrieves and why

**When it runs:** During `plan_probe()`, before any probe executes.

**What it queries:**

```python
# From red_team_agent.py — plan_probe()
query = f"{task.description} target:{task.target_asset}"

for category in (OWASP_GENAI, MITRE_ATTACK, MITRE_ATLAS):
    chunks = retriever.retrieve(query, category=category)
    context_refs.append(chunk.source_id for chunk in chunks)
```

**How it uses the result:**

```
Task: "Test for prompt injection on api.example.com"
        │
        ▼
Query against OWASP_GENAI  →  "LLM01 Prompt Injection techniques..."   source: owasp-genai-v1
Query against MITRE_ATTACK →  "T1190 Exploit Public-Facing Application" source: mitre-attack-v14
Query against MITRE_ATLAS  →  "AML.T0051 LLM Prompt Injection"          source: mitre-atlas-v1
        │
        ▼
ProbePlan.context_refs = ["owasp-genai-v1", "mitre-attack-v14", "mitre-atlas-v1"]
        │
        ▼
Each Finding produced by execute_probe() carries these context_refs
```

**What this means in practice:**

- Ingest `owasp-genai.pdf`, `enterprise-attack.xlsx`, and MITRE ATLAS → every probe plan and finding is traceable back to authoritative adversary knowledge sources
- The `context_refs` on findings tell you exactly which part of OWASP/MITRE informed each discovered weakness

---

### DevSecOps Agent — what it retrieves and why

**When it runs:** During `remediate()`, after confirming the repository is in scope.

**What it queries:**

```python
# From devsecops_agent.py — _open_pr()
chunks = retriever.retrieve(
    query=finding.detail,      # the full vulnerability description
    category=SourceCategory.COMPLIANCE,
)
# Falls back to unfiltered search if compliance yields nothing
```

**How it uses the result:**

```
Finding: "SQL injection vulnerability in /api/users endpoint. Unsanitized input..."
        │
        ▼
Query against COMPLIANCE category:
  →  "OWASP A03 Injection: Use parameterized queries..."   score: 0.94
  →  "CWE-89: Improper Neutralization of SQL..."           score: 0.89
        │
        ▼
guidance_source_ids = ["owasp-compliance-v1", "cwe-v4"]
        │
        ▼
LLM prompt includes: "Secure-coding guidance references: owasp-compliance-v1, cwe-v4"
        │
        ▼
LLM generates a patch grounded in that guidance
        │
        ▼
PR body lists the guidance sources so the reviewer can verify the fix
```

**What this means in practice:**

- Ingest secure-coding guidelines with `--category compliance` → generated patches are grounded in real standards, and the PR reviewer can see exactly which standard informed the fix
- Without compliance ingestion, the LLM still generates a fix but with no grounding references

---

### The full post-ingestion data flow in one diagram

```
Qdrant (populated by ingestion)
  ├── owasp_genai chunks
  ├── mitre_attack chunks
  ├── mitre_atlas chunks
  ├── playbook chunks
  └── compliance chunks
           │
           │   cosine similarity search
           │
    ┌──────┴─────────────────────────────────────┐
    │                                             │
    ▼                                             ▼
Guardrail                                  Blue/Red/DevSecOps
screen_inbound()                           agents
  │                                          │
  │  query: the prompt itself               │  query: finding detail /
  │  categories: owasp, mitre              │           task description
  │  use: score ≥ threshold → block        │  use: chunk.text → LLM context
  │                                         │       chunk.source_id → audit trail
  ▼                                         ▼
BLOCK or FORWARD                     Grounded action + traceable provenance
```

---

### Recommended ingestion order

For a fully operational platform, ingest in this order:

```bash
# 1. Guardrail — attack pattern detection
python -m acdp ingest --source-id owasp-genai-v1  --category owasp_genai  --file owasp-genai.pdf
python -m acdp ingest --source-id mitre-attack-v14 --category mitre_attack --file enterprise-attack.xlsx
python -m acdp ingest --source-id mitre-atlas-v1   --category mitre_atlas  --file atlas.pdf

# 2. Blue Team — containment grounding
python -m acdp ingest --source-id playbook-001 --category playbook --file incident-response.md

# 3. DevSecOps — fix generation grounding
python -m acdp ingest --source-id secure-coding-v1 --category compliance --file secure-coding-guide.pdf
```

Without step 1, the Guardrail has no attack patterns and relies on its blocklist alone.  
Without step 2, Blue Team containment requests have no playbook provenance.  
Without step 3, DevSecOps PRs have no compliance references in their descriptions.

```bash
# Run all unit and property-based tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_blue_team.py -v

# Run only integration tests (requires live Ollama + Qdrant)
pytest -m integration

# Skip integration tests (default)
pytest -m "not integration"
```

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io) and run 100+ iterations per property by default.

---

## 12. Troubleshooting

### `Configuration error: ...`
A required value in `config.yaml` is missing or invalid. The error message names the specific field. Check `config.example.yaml` for the correct format.

### `WARNING: Could not connect to Qdrant`
Qdrant is not running or not reachable at `vector_store_url`. The platform falls back to an in-memory store — data is not persisted across restarts. Start Qdrant with `docker run -p 6333:6333 qdrant/qdrant`.

### `ModelUnavailableError: 'llama3'`
The model is not pulled in Ollama. Run `ollama pull llama3` and `ollama pull nomic-embed-text`.

### `LLMTimeoutError`
A model request exceeded `llm_timeout_seconds`. Increase the timeout in `config.yaml` or use a faster/smaller model.

### `Ingestion error: source file not found`
The `--file` path does not exist. Check the path and file extension.

### `UnicodeDecodeError` on ingest
You are passing a binary file (PDF, Excel) without the correct extension. Ensure your file ends in `.pdf`, `.xlsx`, etc. so the parser selects the right reader.

### `probe_refused` in audit log
The Red Team Agent attempted a probe on an asset that was not in any active scope, or the scope had expired or been revoked. Define a valid scope with `platform.add_scope()` first.

---

## 14. Production Connectors

Production connectors bridge the four agents to live infrastructure. Each connector runs as a standalone long-running process and can be started independently on appropriate hosts.

### Enabling connectors

Add a `connectors:` section to your `config.yaml`. All connectors are disabled by default:

```yaml
connectors:
  guardrail_proxy:
    enabled: true
    host: "0.0.0.0"
    port: 8080
    upstream_url: "http://localhost:11434/api/generate"

  log_stream:
    enabled: true
    mode: tail
    log_path: /var/log/syslog
    scope_id: prod-scope

  scheduler:
    enabled: true
    mode: interval
    interval_seconds: 3600
    targets:
      - "api.example.com"
    auto_remediate: false
    scope_id: prod-scope

  github_pr_client:
    enabled: true
    github_api_base_url: "https://api.github.com"
    max_retries: 3
```

---

### HTTP Proxy / Guardrail API (`python -m acdp serve`)

Starts a FastAPI server that sits inline between users and the upstream LLM. Every prompt is screened by `GuardrailAgent.screen_inbound()` and every response is scrubbed by `GuardrailAgent.scrub_outbound()`.

```bash
# Start with default config discovery (config.yaml → config.example.yaml)
python -m acdp serve

# Start with explicit config path
python -m acdp serve --config /etc/acdp/config.yaml
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat` | Screen prompt, forward to LLM, scrub response |
| `GET`  | `/health`  | Returns `{"status":"ok"}` when operational |

**SSL/TLS configuration:**

```yaml
connectors:
  guardrail_proxy:
    enabled: true
    ssl_certfile: /etc/ssl/certs/acdp.crt
    ssl_keyfile: /etc/ssl/private/acdp.key
```

---

### Log Stream Connector (`python -m acdp monitor`)

Continuously ingests log data and feeds each line to `BlueTeamAgent`. HIGH/CRITICAL findings automatically trigger containment requests.

```bash
python -m acdp monitor --config config.yaml
```

**Ingestion modes:**

| Mode | Config key | Description |
|------|-----------|-------------|
| `tail` | `log_path: /var/log/syslog` | Follow a log file from its current end |
| `syslog_udp` | `udp_host: 0.0.0.0`, `udp_port: 514` | Receive UDP syslog datagrams |
| `poll_dir` | `log_path: /var/log/apps/` | Scan a directory for new/modified files |

**Tail mode example:**

```yaml
connectors:
  log_stream:
    enabled: true
    mode: tail
    log_path: /var/log/auth.log
    retry_interval_seconds: 2.0
    max_retries: 5
    scope_id: prod-scope
```

**Syslog UDP mode example:**

```yaml
connectors:
  log_stream:
    enabled: true
    mode: syslog_udp
    udp_host: "0.0.0.0"
    udp_port: 514
    scope_id: prod-scope
```

**Poll directory mode example:**

```yaml
connectors:
  log_stream:
    enabled: true
    mode: poll_dir
    log_path: /var/log/applications/
    poll_interval_seconds: 10.0
    scope_id: prod-scope
```

---

### Scheduler / Red Team (`python -m acdp scan`)

Runs authorized probe campaigns on a configurable schedule and optionally auto-remediates findings via the DevSecOps Agent.

```bash
python -m acdp scan --config config.yaml
```

**Interval mode:**

```yaml
connectors:
  scheduler:
    enabled: true
    mode: interval
    interval_seconds: 3600      # probe every hour
    targets:
      - "api.example.com"
      - "staging.example.com"
    auto_remediate: false
    scope_id: prod-scope
```

**Cron mode** (five-field standard cron expression, UTC):

```yaml
connectors:
  scheduler:
    enabled: true
    mode: cron
    cron_expression: "0 2 * * 1"   # every Monday at 02:00 UTC
    targets:
      - "api.example.com"
    auto_remediate: true
    scope_id: prod-scope
```

> **Important:** Targets must be within an active `TargetScope` before the scheduler fires, or the probe will be denied and audited. The scheduler will NOT create scopes automatically.

---

### GitHub PR Client

Provides a concrete `PullRequestClient` implementation so the DevSecOps Agent can open real pull requests on GitHub (or GitHub Enterprise) repositories.

**Environment variable:**

```bash
# Set your GitHub personal access token
export GIT_TOKEN=ghp_your_token_here
```

> **Security:** `GIT_TOKEN` must be set as an environment variable. It is never read from `config.yaml` or passed as a constructor argument.

**Configuration:**

```yaml
connectors:
  github_pr_client:
    enabled: true
    github_api_base_url: "https://api.github.com"   # override for GitHub Enterprise
    max_retries: 3                                   # 5xx retry attempts with back-off
```

When `github_pr_client.enabled: true` and `GIT_TOKEN` is set, the platform automatically rebuilds the `DevSecOpsAgent` with the real `GitHubPRClient` on `start_connectors()`.

---

### Running multiple connectors together

You can enable all connectors simultaneously; the Platform starts each one and isolates non-critical failures:

```bash
# All connectors in one process
python -m acdp serve --config config.yaml     # foreground, handles SIGINT
```

Or run each connector in its own process for independent scaling:

```bash
# Terminal 1 — HTTP Proxy
python -m acdp serve --config config.yaml &

# Terminal 2 — Log ingestion
python -m acdp monitor --config config.yaml &

# Terminal 3 — Scheduled probes
python -m acdp scan --config config.yaml &
```

All three processes share the same `AuditLog` file (append-only, concurrent-safe) and will write their decisions there.

---

### Graceful shutdown

All three CLI commands handle `SIGINT` (Ctrl+C) and `SIGTERM` gracefully:

1. In-flight log lines or probe results are flushed
2. Open file handles and UDP sockets are closed
3. The process exits with code `0`

On configuration error (bad config file), the process exits with code `1` and prints a descriptive message to `stderr` without printing the startup banner.

---

## 15. Running Tests

```bash
# Run all unit and property-based tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_connectors.py -v

# Run only integration tests (requires live Ollama + Qdrant + GitHub)
pytest -m integration

# Skip integration tests (default)
pytest -m "not integration"
```

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io) and run 100+ iterations per property by default.
