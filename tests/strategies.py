"""Shared Hypothesis strategies for the Autonomous Cyber Defense Platform.

This module is intentionally a stub. Each implementation phase adds the
generators it needs, and they accumulate here so property-based tests across
the platform can share a single, consistent set of strategies. Examples of
strategies that later tasks will add:

- ``platform_configs()``            (config round-trip / defaults)
- ``findings()``                    (finding identity)
- ``knowledge_sources()`` / ``embedded_chunks()``  (ingestion)
- ``target_scopes()`` / ``action_requests()`` / ``eval_times()``  (authorization)
- ``prompts_with_scores()``         (guardrail inbound)
- ``text_with_pii_and_secrets()``   (guardrail outbound)
- ``telemetry_records()`` / ``normalized_events()``  (telemetry round-trip)
- ``security_events()`` / ``task_plans()``  (orchestrator routing)

Strategies live here (rather than inline in test modules) so they can be
reused and composed as the platform grows.
"""

from __future__ import annotations

from hypothesis import strategies as st

from datetime import datetime, timezone

from acdp.models import (
    ActionRequest,
    AuditAction,
    AuditRecord,
    EmbeddedChunk,
    Finding,
    KnowledgeSource,
    NormalizedEvent,
    PlatformConfig,
    SecurityEvent,
    SecurityEventType,
    Severity,
    SourceCategory,
    Task,
    TaskPlan,
    TaskState,
    TargetScope,
)

__all__ = [
    "platform_configs",
    "partial_platform_config_dicts",
    "audit_records",
    "findings",
    "model_names",
    "knowledge_sources",
    "embedded_chunks",
    "retrieval_scenarios",
    "category_filtered_retrieval_scenarios",
    "target_scopes",
    "action_requests",
    "eval_times",
    "prompts_with_scores",
    "text_with_pii_and_secrets",
    "clean_text",
    "telemetry_records",
    "normalized_events",
    "probe_plans_with_scopes",
    "security_events",
    "task_plans",
]


# A small, YAML-safe alphabet for string-valued settings. Kept printable and
# free of control characters so values survive a YAML dump/load round-trip
# unchanged.
_text = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=0,
    max_size=40,
)

# Finite floats only: NaN/inf are not representable in round-trippable YAML and
# are not valid platform settings.
_finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e9,
    max_value=1e9,
)


@st.composite
def platform_configs(draw: st.DrawFn) -> PlatformConfig:
    """Generate valid :class:`PlatformConfig` instances with all fields populated.

    Every field is explicitly drawn (rather than relying on model defaults) so
    that round-trip / serialization properties are exercised across the full
    configuration space, not just the default configuration.
    """
    return PlatformConfig(
        reasoning_model=draw(_text),
        embedding_model=draw(_text),
        llm_timeout_seconds=draw(_finite_floats),
        # similarity_threshold is a guardrail configurable in [0, 1) (Req 5.2).
        similarity_threshold=draw(
            st.floats(min_value=0.0, max_value=1.0, exclude_max=True)
        ),
        top_k=draw(st.integers(min_value=0, max_value=1000)),
        severity_threshold=draw(st.sampled_from(list(Severity))),
        containment_requires_approval=draw(st.booleans()),
        guardrail_default_action=draw(st.sampled_from(["allow", "block"])),
        vector_store_url=draw(_text),
        ollama_url=draw(_text),
        audit_log_path=draw(_text),
    )


# Per-field value strategies, expressed in the JSON/YAML-scalar form that a
# configuration file would actually contain (enums as their string values,
# literals as strings). These are used to build *partial* configuration
# mappings where any subset of settings may be omitted.
_field_value_strategies: dict[str, st.SearchStrategy[object]] = {
    "reasoning_model": _text,
    "embedding_model": _text,
    "llm_timeout_seconds": _finite_floats,
    "similarity_threshold": st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
    "top_k": st.integers(min_value=0, max_value=1000),
    "severity_threshold": st.sampled_from([s.value for s in Severity]),
    "containment_requires_approval": st.booleans(),
    "guardrail_default_action": st.sampled_from(["allow", "block"]),
    "vector_store_url": _text,
    "ollama_url": _text,
    "audit_log_path": _text,
}


@st.composite
def partial_platform_config_dicts(draw: st.DrawFn) -> dict[str, object]:
    """Generate a *partial* config mapping that omits at least one optional value.

    Every :class:`PlatformConfig` field is optional (each has a documented
    default), so a partial mapping is any strict subset of the settings. The
    returned dict is YAML-safe (enums/literals expressed as their scalar
    string form) and is guaranteed to omit at least one setting, so that the
    "apply documented default" behaviour (Req 13.5) is always exercised.

    Callers can recover the set of omitted settings as
    ``set(_field_value_strategies) - set(result)``.
    """
    field_names = list(_field_value_strategies)
    # Choose the settings to *include*; force at least one omission by keeping
    # the included set a strict subset of all fields.
    included = draw(
        st.lists(
            st.sampled_from(field_names),
            min_size=0,
            max_size=len(field_names) - 1,
            unique=True,
        )
    )
    return {name: draw(_field_value_strategies[name]) for name in included}


@st.composite
def audit_records(draw: st.DrawFn) -> AuditRecord:
    """Generate well-formed :class:`AuditRecord` instances for audit-log tests.

    Every record carries the four fields Req 12.1 requires — a timestamp, an
    actor identifier, an action type, and an outcome — plus the optional
    ``target``/``detail`` fields. ``seq`` is deliberately left ``None`` because
    the sequence id is assigned by the log on append (not by the caller).
    """
    return AuditRecord(
        timestamp=draw(st.datetimes()),
        actor_id=draw(
            st.one_of(
                st.just("orchestrator"),
                _text.filter(lambda s: s != ""),
            )
        ),
        action=draw(st.sampled_from(list(AuditAction))),
        outcome=draw(_text),
        target=draw(st.one_of(st.none(), _text)),
        detail=draw(
            st.dictionaries(
                keys=_text.filter(lambda s: s != ""),
                values=_text,
                max_size=4,
            )
        ),
    )


# Identifier alphabet for finding ids and originating-event ids: non-empty,
# printable tokens with no surrounding whitespace so identity comparisons are
# unambiguous. Constrained to a small alphabet so that the ``unique_by`` filter
# below still produces useful collisions/spread across many examples.
_identifier = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
    min_size=1,
    max_size=24,
)


@st.composite
def findings(draw: st.DrawFn) -> Finding:
    """Generate a valid :class:`Finding` with a non-empty originating-event link.

    Every generated finding carries a non-empty ``finding_id`` and a non-empty
    ``originating_event_id`` so the identity contract (Req 12.3) — unique id
    per finding, each linked to its originating event — is meaningfully
    exercised. ``finding_id`` uniqueness across a *set* of findings is enforced
    at the collection level (see ``finding_lists``) rather than per-instance.
    """
    return Finding(
        finding_id=draw(_identifier),
        originating_event_id=draw(_identifier),
        agent_id=draw(_identifier),
        severity=draw(st.sampled_from(list(Severity))),
        title=draw(_text),
        detail=draw(_text),
        asset=draw(st.one_of(st.none(), _text)),
        context_refs=draw(st.lists(_identifier, max_size=4)),
        created_at=draw(st.datetimes()),
    )


# Model-name alphabet for LLM Gateway routing tests. Ollama model names are
# non-empty printable tokens (e.g. "llama3", "nomic-embed-text:latest"); we
# constrain to a small alphabet so distinct requested/configured names collide
# meaningfully across many examples while staying valid, non-empty identifiers.
_model_name = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
    min_size=1,
    max_size=32,
)


def model_names() -> st.SearchStrategy[str]:
    """Generate non-empty LLM model names for gateway routing properties.

    A model name is any non-empty printable token, matching the free-form model
    identifiers accepted by the LLM Gateway (Req 4.4). Kept small so that the
    routing property meaningfully exercises many distinct requested names.
    """
    return _model_name


# Knowledge-source content alphabet. Kept printable and free of control
# characters so chunking (which splits on whitespace) behaves predictably, and
# includes ASCII whitespace so generated content spans multiple whitespace-
# delimited tokens — enough to exercise the chunker's boundary packing.
_source_content = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=0,
    max_size=2000,
)


@st.composite
def knowledge_sources(draw: st.DrawFn) -> KnowledgeSource:
    """Generate :class:`KnowledgeSource` instances for ingestion properties.

    Each source carries a non-empty ``source_id``, a drawn
    :class:`~acdp.models.SourceCategory` (so the category-tagging contract in
    Req 2.3 is exercised across every category, including the MITRE/OWASP ones),
    and content that contains at least one non-whitespace token. Content with
    only whitespace is filtered out because it is *unreadable* (the ingestion
    pipeline rejects it with :class:`~acdp.exceptions.IngestionError`); the
    "stored and tagged" property (Req 2.1-2.3) concerns readable sources, so we
    constrain the generator to readable content.
    """
    return KnowledgeSource(
        source_id=draw(_identifier),
        category=draw(st.sampled_from(list(SourceCategory))),
        content=draw(_source_content.filter(lambda s: s.split())),
    )


# Embedding vector strategy: finite floats in a bounded range, fixed to a small
# non-zero dimension so vectors are comparable and cosine similarity is
# well-defined. A minimum size of 1 keeps every vector non-empty.
_embedding_vector = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    min_size=1,
    max_size=8,
)


@st.composite
def embedded_chunks(draw: st.DrawFn) -> EmbeddedChunk:
    """Generate well-formed :class:`EmbeddedChunk` instances for store tests.

    Every chunk carries the tagging fields the storage contract depends on — a
    ``chunk_id``, a ``source_id``, a :class:`~acdp.models.SourceCategory`, and an
    ingestion timestamp (Req 2.2, 2.3) — plus a non-empty embedding vector.
    """
    return EmbeddedChunk(
        chunk_id=draw(_identifier),
        source_id=draw(_identifier),
        category=draw(st.sampled_from(list(SourceCategory))),
        ingested_at=draw(st.datetimes()),
        text=draw(_text),
        vector=draw(_embedding_vector),
    )


# A single finite-float coordinate for embedding/query vectors. Bounded so
# cosine similarity stays numerically well-behaved across many examples.
_vector_coordinate = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
)


@st.composite
def retrieval_scenarios(draw: st.DrawFn) -> dict[str, object]:
    """Generate a populated-store retrieval scenario for the top-K property.

    Produces a self-consistent bundle for exercising the retriever end to end:

    * ``query_vector`` — the vector the (stubbed) gateway will return for the
      query, and
    * ``chunks`` — a non-empty list of :class:`~acdp.models.EmbeddedChunk` with
      **unique** ``chunk_id`` values (so an in-memory upsert stores every one
      rather than collapsing by key) whose vectors all share the query's
      dimension (so cosine similarity is well-defined), and
    * ``top_k`` — a configurable K spanning below, at, and above the number of
      stored chunks so the "at most K" bound is meaningfully tested (Req 3.1).

    All vectors share one drawn dimension so query and chunk vectors are
    comparable; the store is guaranteed populated (at least one chunk).
    """
    dimension = draw(st.integers(min_value=1, max_value=6))
    vector = st.lists(
        _vector_coordinate, min_size=dimension, max_size=dimension
    )

    query_vector = draw(vector)

    n_chunks = draw(st.integers(min_value=1, max_value=8))
    # Unique chunk_ids so every generated chunk survives an upsert keyed by
    # chunk_id (otherwise same-id chunks would replace one another).
    chunk_ids = draw(
        st.lists(_identifier, min_size=n_chunks, max_size=n_chunks, unique=True)
    )
    chunks = [
        EmbeddedChunk(
            chunk_id=chunk_id,
            source_id=draw(_identifier),
            category=draw(st.sampled_from(list(SourceCategory))),
            ingested_at=draw(st.datetimes()),
            text=draw(_text),
            vector=draw(vector),
        )
        for chunk_id in chunk_ids
    ]

    # Span K from below to above the stored-chunk count so we exercise
    # truncation (K < n), exact fit (K == n), and over-ask (K > n).
    top_k = draw(st.integers(min_value=1, max_value=n_chunks + 3))

    return {"query_vector": query_vector, "chunks": chunks, "top_k": top_k}


@st.composite
def category_filtered_retrieval_scenarios(draw: st.DrawFn) -> dict[str, object]:
    """Generate a retrieval scenario paired with a source-category filter.

    Produces a self-consistent bundle for exercising the retriever's
    category-filter contract (Req 3.3):

    * ``query_vector`` — the vector the (stubbed) gateway returns for the query,
    * ``chunks`` — a non-empty list of :class:`~acdp.models.EmbeddedChunk` with
      **unique** ``chunk_id`` values (so every chunk survives an upsert keyed by
      chunk_id) whose vectors all share the query's dimension (so cosine
      similarity is well-defined). Chunk categories are drawn freely across the
      full :class:`~acdp.models.SourceCategory` set so that, for any drawn
      ``category``, the store may contain zero, some, or all matching chunks,
      and
    * ``category`` — the source-category filter to apply, drawn independently of
      the stored chunks' categories so that both matching and non-matching
      (empty-result) cases are exercised.

    All vectors share one drawn dimension so query and chunk vectors are
    comparable; the store is guaranteed populated (at least one chunk).
    """
    dimension = draw(st.integers(min_value=1, max_value=6))
    vector = st.lists(_vector_coordinate, min_size=dimension, max_size=dimension)

    query_vector = draw(vector)

    n_chunks = draw(st.integers(min_value=1, max_value=8))
    chunk_ids = draw(
        st.lists(_identifier, min_size=n_chunks, max_size=n_chunks, unique=True)
    )
    chunks = [
        EmbeddedChunk(
            chunk_id=chunk_id,
            source_id=draw(_identifier),
            category=draw(st.sampled_from(list(SourceCategory))),
            ingested_at=draw(st.datetimes()),
            text=draw(_text),
            vector=draw(vector),
        )
        for chunk_id in chunk_ids
    ]

    # The filter category is drawn independently of the chunks' categories so
    # the returned set spans none/some/all matches across many examples.
    category = draw(st.sampled_from(list(SourceCategory)))

    top_k = draw(st.integers(min_value=1, max_value=n_chunks + 3))

    return {
        "query_vector": query_vector,
        "chunks": chunks,
        "category": category,
        "top_k": top_k,
    }


@st.composite
def target_scopes(draw: st.DrawFn) -> TargetScope:
    """Generate :class:`TargetScope` instances for the scope-registry properties.

    Every scope carries a non-empty ``scope_id``, a (possibly empty) list of
    asset identifiers, a ``created_at`` timestamp, a required ``expires_at``
    expiration (Req 11.3), and a drawn ``revoked`` flag so that both active and
    inactive (revoked) scopes are exercised. ``created_at`` and ``expires_at``
    are drawn independently so the generator spans scopes that are already
    expired, currently valid, and not-yet-created relative to any evaluation
    time.

    Assets are drawn from a small shared pool (:data:`_authz_asset`) and
    timestamps from a narrow shared window (:data:`_authz_time`) so that, when
    a scope is paired with an independently generated
    :func:`action_requests` / :func:`eval_times` value, the asset and the
    evaluation time collide often enough to exercise **both** the GRANT case
    (asset in an active scope) and the several DENY cases (out of scope,
    expired, revoked) across the example budget — rather than degenerating to
    an almost-always-DENY generator.
    """
    return TargetScope(
        scope_id=draw(_identifier),
        assets=draw(st.lists(_authz_asset, max_size=5)),
        created_at=draw(_authz_time),
        expires_at=draw(_authz_time),
        revoked=draw(st.booleans()),
    )


# --- Authorization strategies (Property 1) ---------------------------------
#
# The fail-closed authorization property pairs an ``ActionRequest`` (a target
# asset), a set of ``TargetScope`` records (mixed active/expired/revoked), and
# an evaluation time. For the property to meaningfully exercise GRANT as well
# as the several DENY branches, the request's asset and the evaluation time
# must collide with scope assets and scope validity windows often enough. We
# therefore draw scope assets, request assets, and all timestamps from small
# *shared* pools so overlaps happen by construction.

# A small shared pool of asset identifiers. Because both scopes and requests
# draw from the same handful of names, a request's asset frequently *is* a
# member of some scope — so GRANT and out-of-scope DENY are both common.
_authz_asset = st.sampled_from(
    ["host-a", "host-b", "host-c", "repo-x", "repo-y", "endpoint-1", "domain.test"]
)

# A narrow shared window of timestamps (a single day, at hour granularity).
# Scope ``created_at``/``expires_at`` and the evaluation time all draw from
# this window, so a scope is expired at the eval time roughly as often as it is
# still valid — exercising the expired-scope DENY branch alongside GRANT.
_authz_time = st.datetimes(
    min_value=datetime(2024, 1, 1, 0, 0, 0),
    max_value=datetime(2024, 1, 2, 0, 0, 0),
    timezones=st.just(timezone.utc),
)


@st.composite
def action_requests(draw: st.DrawFn) -> ActionRequest:
    """Generate :class:`ActionRequest` instances for the authorization property.

    Each request names a requesting ``agent_id``, a target ``asset`` drawn from
    the shared :data:`_authz_asset` pool (so it collides with scope assets often
    enough to exercise GRANT), and a free-form ``action`` label drawn from the
    kinds of side effect the platform gates — ``"probe"`` (Red Team),
    ``"containment"`` (Blue Team), and ``"pr"`` (DevSecOps) — so the audited
    action kind spans the real callers (Req 11.1, 11.5).
    """
    return ActionRequest(
        agent_id=draw(_identifier),
        asset=draw(_authz_asset),
        action=draw(st.sampled_from(["probe", "containment", "pr", "act"])),
    )


def eval_times() -> st.SearchStrategy[datetime]:
    """Generate evaluation timestamps for the authorization property.

    Drawn from the same narrow shared window as scope timestamps
    (:data:`_authz_time`) so that, relative to a generated scope, the
    evaluation time falls before expiry (scope valid) about as often as after
    (scope expired) — exercising both the GRANT and expired-scope DENY
    branches (Property 1).
    """
    return _authz_time


# --- Guardrail inbound strategies (Property 11) ----------------------------
#
# The inbound-screening property blocks a prompt iff the *maximum* injected
# similarity score (across the blocklist patterns and retrieved attack
# patterns) is at or above the configured threshold. To exercise both the BLOCK
# and FORWARD branches across the example budget, the threshold and the scores
# are drawn from the same [0, 1] range so a given score straddles the threshold
# often, rather than degenerating to an almost-always-forward generator.

# Prompt text for screening. Printable, control-character-free, and possibly
# empty — screening a prompt is a pure function of the injected scores, so the
# prompt content only needs to survive a byte-for-byte "forward unchanged"
# comparison (Req 5.3).
_prompt_text = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=0,
    max_size=120,
)

# A single injected similarity score in [0, 1]. Bounded to the closed unit
# interval so scores straddle any threshold in [0, 1) — exercising the
# at-or-above (block) and below (forward) branches (Req 5.2).
_similarity_score = st.floats(
    allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0
)


@st.composite
def prompts_with_scores(draw: st.DrawFn) -> dict[str, object]:
    """Generate an inbound prompt with injected similarity scores for Property 11.

    Produces a self-consistent bundle for exercising
    :meth:`~acdp.agents.guardrail.GuardrailAgent.screen_inbound` deterministically,
    with the embedding/similarity step *injected* rather than computed:

    * ``prompt`` — the inbound prompt text (possibly empty), returned
      byte-for-byte unchanged on a forward (Req 5.3);
    * ``threshold`` — the configured similarity threshold, drawn in ``[0, 1)``
      (the operator may configure any value below 1.0, Req 5.2);
    * ``scores`` — a list of ``(pattern_id, similarity)`` pairs the stub
      Blocklist will return for this prompt (Req 5.1). ``pattern_id`` values are
      unique so the pattern responsible for a block is unambiguous; the list may
      be empty (no pattern matched at all).

    Scores and threshold share the ``[0, 1]`` range so, across the example
    budget, the maximum score falls at/above the threshold (BLOCK) about as
    often as below it (FORWARD).
    """
    prompt = draw(_prompt_text)
    # Threshold strictly below 1.0 — the operator-configurable range (Req 5.2).
    threshold = draw(st.floats(min_value=0.0, max_value=1.0, exclude_max=True))

    n_scores = draw(st.integers(min_value=0, max_value=6))
    pattern_ids = draw(
        st.lists(_identifier, min_size=n_scores, max_size=n_scores, unique=True)
    )
    scores = [(pattern_id, draw(_similarity_score)) for pattern_id in pattern_ids]

    return {"prompt": prompt, "threshold": threshold, "scores": scores}


# --- Guardrail outbound strategies (Property 12) ---------------------------
#
# The outbound-scrubbing property asserts that every *detected* PII value and
# secret value is removed from the returned text. To exercise that meaningfully
# and deterministically, we generate text that embeds values matching the
# Guardrail Agent's detectors (see ``acdp.agents.guardrail._DETECTORS``): PII
# (emails, SSNs, and financial account numbers) and secrets (API keys).
#
# Generated sensitive values are always surrounded by *letter-only* filler
# words separated by single spaces. This is a deliberate, input-space-aware
# constraint: the letter fillers guarantee the word boundaries the detectors
# require fire, and they break up adjacent numeric runs so no two injected
# values are ever merged (or split) by a single regex match. The filler
# alphabet is lowercase letters only, so filler can never itself form a PII or
# secret pattern, nor accidentally contain an injected value as a substring.

# Digit strategy for the numeric components of PII values.
_digit = st.integers(min_value=0, max_value=9).map(str)


def _digits(n: int) -> st.SearchStrategy[str]:
    """A strategy for an ``n``-digit numeric string (leading zeros allowed)."""
    return st.lists(_digit, min_size=n, max_size=n).map("".join)


# Letter-only filler words. Non-empty so they always separate adjacent values;
# lowercase ASCII letters only so filler can never form a PII/secret pattern.
_filler_word = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1,
    max_size=10,
)

# Alphanumeric token component (letters + digits), used for email local/domain
# parts and API-key bodies.
_alnum = st.text(
    alphabet=st.characters(
        whitelist_categories=(),
        whitelist_characters=(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    ),
    min_size=1,
    max_size=12,
)


@st.composite
def _emails(draw: st.DrawFn) -> str:
    """Generate an email matching the Guardrail EMAIL detector (Req 6.2)."""
    local = draw(_alnum)
    domain = draw(_alnum)
    tld = draw(
        st.text(
            alphabet=st.characters(
                min_codepoint=ord("a"), max_codepoint=ord("z")
            ),
            min_size=2,
            max_size=4,
        )
    )
    return f"{local}@{domain}.{tld}"


@st.composite
def _ssns(draw: st.DrawFn) -> str:
    """Generate a US SSN (NNN-NN-NNNN) matching the SSN detector (Req 6.2)."""
    return f"{draw(_digits(3))}-{draw(_digits(2))}-{draw(_digits(4))}"


@st.composite
def _financial_accounts(draw: st.DrawFn) -> str:
    """Generate an 8-12 digit financial account number (Req 6.2).

    Kept strictly below the 13-digit payment-card minimum so the value is
    attributed to the financial-account detector rather than the card detector;
    either way it is masked, but this keeps the generated category unambiguous.
    """
    n = draw(st.integers(min_value=8, max_value=12))
    return draw(_digits(n))


@st.composite
def _api_keys(draw: st.DrawFn) -> str:
    """Generate a secret API key matching one of the API_KEY detector forms (Req 6.3)."""
    body = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=(),
                whitelist_characters=(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),
            ),
            min_size=16,
            max_size=32,
        )
    )
    prefix = draw(st.sampled_from(["sk", "pk", "rk", "api", "key", "token"]))
    sep = draw(st.sampled_from(["-", "_"]))
    infix = draw(st.sampled_from(["", "live", "test", "prod"]))
    # Provider-prefixed high-entropy token, e.g. ``sk-live_ABC...`` or ``api_ABC...``.
    provider_style = f"{prefix}{sep}{infix}{body}" if infix else f"{prefix}{sep}{body}"
    # AWS-style access key id: ``AKIA`` + 16 upper/digit chars.
    aws_body = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=(),
                whitelist_characters="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            ),
            min_size=16,
            max_size=16,
        )
    )
    aws_style = f"AKIA{aws_body}"
    return draw(st.sampled_from([provider_style, aws_style]))


# PII value generators (Req 6.2): emails, SSNs, and financial account numbers.
_pii_value = st.one_of(_emails(), _ssns(), _financial_accounts())

# Secret value generators (Req 6.3): API keys.
_secret_value = _api_keys()


@st.composite
def text_with_pii_and_secrets(draw: st.DrawFn) -> dict[str, object]:
    """Generate outbound text embedding detectable PII and secret values (Property 12).

    Produces a self-consistent bundle for exercising
    :meth:`~acdp.agents.guardrail.GuardrailAgent.scrub_outbound`:

    * ``text`` — an outbound response containing at least one PII value (email,
      SSN, or financial account number, Req 6.2) **and** at least one secret
      value (API key, Req 6.3), each surrounded by letter-only filler words so
      the detectors' word boundaries fire and no two values are ever merged or
      split by a single regex match; and
    * ``sensitive_values`` — the exact raw substrings injected into ``text``,
      so the property can assert every one of them is absent from the scrubbed
      output.

    At least one PII value and one secret value are always present, so scrubbing
    is guaranteed to occur — exercising the "removes every detected value"
    contract (Req 6.1-6.3) rather than the clean-passthrough case (Property 13).
    """
    pii_values: list[str] = draw(
        st.lists(_pii_value, min_size=1, max_size=4)
    )
    secret_values: list[str] = draw(
        st.lists(_secret_value, min_size=1, max_size=3)
    )

    # Interleave the sensitive values in a random order so PII and secrets are
    # not positionally correlated.
    values = draw(st.permutations(pii_values + secret_values))

    # Build "word value word value ... word": every value is flanked by
    # letter-only filler words, guaranteeing clean boundaries between values.
    parts: list[str] = []
    for value in values:
        parts.append(draw(_filler_word))
        parts.append(value)
    parts.append(draw(_filler_word))
    text = " ".join(parts)

    return {"text": text, "sensitive_values": list(values)}


# --- Guardrail outbound clean-passthrough strategy (Property 13) -----------
#
# The clean-passthrough property asserts that an outbound response containing
# *no* PII and *no* secret patterns is returned byte-for-byte unchanged with no
# masking/redaction applied and no audit record written (Req 6.5). To exercise
# that meaningfully we generate text that is guaranteed to contain none of the
# values the Guardrail Agent's detectors match.
#
# Rather than hand-restricting the alphabet (which is fragile — the detectors
# span emails, SSNs, card/account digit-runs, and several API-key forms), we
# generate free-form printable text and then *filter it through the agent's own
# detectors*, keeping only samples that no detector matches. This ties the
# generator directly to ``acdp.agents.guardrail._DETECTORS`` so it stays clean
# by construction even if the detector set evolves.


def _has_no_detectable_sensitive_value(text: str) -> bool:
    """Return ``True`` iff none of the Guardrail Agent's detectors match ``text``.

    Used as a Hypothesis filter so ``clean_text`` only yields responses that the
    scrubber would leave untouched (Req 6.5). Imported lazily to avoid a
    module-level import cycle between the strategies module and the agent.
    """
    from acdp.agents.guardrail_agent import _DETECTORS

    return not any(pattern.search(text) for _category, pattern in _DETECTORS)


# Free-form printable text (control-character-free), possibly empty. Filtered
# below so only genuinely clean samples survive.
_clean_candidate_text = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=0,
    max_size=200,
)


def clean_text() -> st.SearchStrategy[str]:
    """Generate outbound text containing no detectable PII or secret values (Property 13).

    Produces free-form printable responses that the Guardrail Agent's detectors
    (:data:`acdp.agents.guardrail._DETECTORS`) leave completely untouched — no
    email, SSN, payment-card/financial-account digit run, or API-key pattern.
    Because the samples are filtered through the agent's *actual* detector set,
    every generated string is guaranteed "clean": scrubbing it must return it
    byte-for-byte unchanged with no masking and no audit record (Req 6.5).

    The generator spans the empty string and multi-word printable text, so the
    passthrough property is exercised across a broad slice of the clean input
    space rather than a hand-picked constant.
    """
    return _clean_candidate_text.filter(_has_no_detectable_sensitive_value)


# --- Telemetry strategies (Property 15) ------------------------------------
#
# The telemetry round-trip property (Req 7.1, 7.3, 7.4) needs:
#   1. ``telemetry_records()`` — raw strings in a *supported* format (JSON,
#      syslog RFC 3164, CEF) that BlueTeamAgent.parse() will accept without
#      raising TelemetryParseError, and
#   2. ``normalized_events()`` — NormalizedEvent instances that are
#      consistent with what parse() would produce so that serialize() →
#      parse() round-trips them faithfully.
#
# Both generators constrain their output to the intersection of what the
# serializer *emits* and the parser *accepts* — the round-trip-safe subset.
# In particular:
#   - JSON: all field values are plain strings, event_id is non-empty, the
#     timestamp is always ISO-8601 with explicit UTC offset, and extra
#     attribute keys avoid the reserved field names.
#   - syslog (RFC 3164): only the fields the serializer writes — month, day,
#     time, host, tag, pid, msg — and only values that survive a serialize →
#     parse cycle (no leading/trailing whitespace in tags, no colon in tag).
#   - CEF: only the fields the serializer writes — vendor, product, dev_version,
#     sig_id, name, severity, plus extension key=value pairs. Keys and values
#     avoid characters that break the CEF extension tokeniser (``=``, ``|``).

# Alphabet for identifiers safe in all three formats: alphanumeric + hyphen.
# No spaces, ``|``, ``=``, ``:``, or control characters that would confuse
# syslog/CEF parsers.
_tel_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=20,
)

# Attribute key alphabet: lowercase letters + underscore, non-empty.
_tel_attr_key = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1,
    max_size=12,
).filter(lambda k: k not in {
    "event_id", "timestamp", "ts", "host", "actor",
    "action", "severity", "source_format",
    # CEF reserved attr keys kept by the serializer:
    "vendor", "product", "dev_version", "sig_id", "cef_version",
    # syslog reserved attr keys:
    "tag", "msg", "pid", "pri", "version", "app", "procid", "msgid",
    # CEF extension keys that are interpreted as host/actor by the parser:
    "dhost", "src", "suser", "duser", "rt", "start", "end", "act",
})

# Attribute value alphabet: printable ASCII, no ``=`` or ``|`` (would break
# CEF extension parsing) and no ``\r``/``\n`` (would break syslog line parsing).
_tel_attr_val = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E).filter(
        lambda c: c not in "=|"
    ),
    min_size=0,
    max_size=30,
)

# A fixed UTC timestamp in ISO-8601 format — always round-trips cleanly
# through JSON serialize → parse (the serializer uses .isoformat() which the
# parser accepts via datetime.fromisoformat).
_tel_timestamp_str = "2024-06-15T12:00:00+00:00"
_tel_timestamp_dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

# Host and actor values: non-empty, safe in syslog and CEF (no whitespace or
# special characters).
_tel_host = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-.",
    min_size=1,
    max_size=20,
)
_tel_actor = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_.",
    min_size=1,
    max_size=20,
)
# Action values: safe in CEF (used as the CEF ``name`` field) — no ``|``.
_tel_action = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_ ",
    min_size=1,
    max_size=20,
)


@st.composite
def telemetry_records(draw: st.DrawFn) -> tuple[str, str]:
    """Generate a (raw_telemetry, source_format) pair for Property 15.

    Produces a raw string in one of the three supported formats — ``"json"``,
    ``"syslog"``, or ``"cef"`` — that :meth:`BlueTeamAgent.parse` will accept
    without raising :class:`~acdp.agents.blue.TelemetryParseError`, and
    returns the format tag alongside the raw string so the test can construct
    the round-trip assertions without re-detecting the format.

    The generated values are constrained to the *round-trip-safe* subset of
    each format: only field values that the serializer writes *and* the parser
    reads back unchanged, so that parse → serialize → parse produces equivalent
    events (Req 7.4).
    """
    fmt = draw(st.sampled_from(["json", "syslog", "cef"]))

    if fmt == "json":
        event_id = draw(_tel_id)
        host = draw(st.one_of(st.none(), _tel_host))
        actor = draw(st.one_of(st.none(), _tel_actor))
        action = draw(st.one_of(st.none(), _tel_action))
        severity = draw(st.sampled_from(list(Severity)))
        extra_attrs = draw(
            st.dictionaries(
                keys=_tel_attr_key,
                values=_tel_attr_val,
                max_size=4,
            )
        )
        data: dict[str, object] = {
            "event_id": event_id,
            "timestamp": _tel_timestamp_str,
            "severity": severity.value,
        }
        if host is not None:
            data["host"] = host
        if actor is not None:
            data["actor"] = actor
        if action is not None:
            data["action"] = action
        data.update(extra_attrs)
        import json as _json
        return _json.dumps(data, sort_keys=True), "json"

    if fmt == "syslog":
        # Generate RFC 3164: "Mon DD HH:MM:SS host tag[pid]: msg"
        # Use a fixed date/time to avoid year-inference edge cases at year
        # boundaries and to keep the round-trip timestamp stable.
        host = draw(_tel_host)
        # Tag: alphanumeric + hyphen/underscore, no colon, no brackets.
        tag = draw(st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
            min_size=1,
            max_size=12,
        ))
        pid = draw(st.one_of(
            st.none(),
            st.integers(min_value=1, max_value=99999).map(str),
        ))
        # msg: printable ASCII, no newlines.
        msg = draw(st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E).filter(
                lambda c: c not in "\r\n"
            ),
            min_size=0,
            max_size=60,
        ))
        # Fixed date: Jun 15 12:00:00 (matches _tel_timestamp_dt).
        tag_part = f"{tag}[{pid}]" if pid else tag
        raw = f"Jun 15 12:00:00 {host} {tag_part}: {msg}"
        return raw, "syslog"

    # fmt == "cef"
    vendor = draw(_tel_host)   # reuse safe alphabet
    product = draw(_tel_host)
    dev_version = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789._-",
        min_size=1, max_size=10,
    ))
    sig_id = draw(_tel_id)
    name = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _-",
        min_size=1, max_size=20,
    ))
    severity = draw(st.sampled_from(list(Severity)))
    cef_sev_map = {
        Severity.INFO: "0",
        Severity.LOW: "3",
        Severity.MEDIUM: "5",
        Severity.HIGH: "8",
        Severity.CRITICAL: "10",
    }
    cef_sev = cef_sev_map[severity]
    # Extension: key=value pairs; keys and values safe for CEF tokenizer.
    ext_attrs = draw(
        st.dictionaries(
            keys=_tel_attr_key,
            values=_tel_attr_val,
            max_size=3,
        )
    )
    ext_str = " ".join(f"{k}={v}" for k, v in sorted(ext_attrs.items()))
    raw = f"CEF:0|{vendor}|{product}|{dev_version}|{sig_id}|{name}|{cef_sev}|{ext_str}"
    return raw, "cef"


@st.composite
def normalized_events(draw: st.DrawFn) -> NormalizedEvent:
    """Generate :class:`NormalizedEvent` instances for telemetry round-trip tests.

    Every generated event uses field values that are in the *round-trip-safe*
    subset of the JSON serializer/parser pair: the event is serialized to JSON
    by :meth:`BlueTeamAgent.serialize` and re-parsed by
    :meth:`BlueTeamAgent.parse`, so the round-trip produces an equivalent
    event (Req 7.4).  JSON is used as the canonical format here because its
    serialize → parse cycle preserves all NormalizedEvent fields deterministically
    (event_id, host, actor, action, severity, and attributes).
    """
    event_id = draw(_tel_id)
    host = draw(st.one_of(st.none(), _tel_host))
    actor = draw(st.one_of(st.none(), _tel_actor))
    action = draw(st.one_of(st.none(), _tel_action.filter(lambda s: s.strip() == s)))
    severity = draw(st.sampled_from(list(Severity)))
    attributes = draw(
        st.dictionaries(
            keys=_tel_attr_key,
            values=_tel_attr_val,
            max_size=4,
        )
    )
    return NormalizedEvent(
        event_id=event_id,
        source_format="json",
        timestamp=_tel_timestamp_dt,
        host=host,
        actor=actor,
        action=action,
        severity=severity,
        attributes=dict(sorted(attributes.items())),
    )


# --- Red Team Agent strategies (Property 21) --------------------------------
#
# Property 21 asserts that execute_probe reports exactly one Finding per
# weakness in the plan when the probe is authorized. The generator needs to
# produce:
#   - a ProbePlan with an arbitrary (possibly empty) list of weaknesses, and
#   - a TargetScope that is active and contains the plan's target_asset,
#     so the AuthorizationService always grants the probe and the one-to-one
#     mapping is exercised — not the refusal path.
#
# Weakness strings are drawn from a short printable alphabet to keep them
# valid while spanning the 0-to-N range.

_weakness_text = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
    min_size=1,
    max_size=40,
)

# A fixed future timestamp used for scope expiration so the scope is always
# active during the test (avoids coupling the generator to wall-clock time).
_RED_TEAM_EXPIRES_AT = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
_RED_TEAM_CREATED_AT = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


@st.composite
def probe_plans_with_scopes(draw: st.DrawFn) -> dict[str, object]:
    """Generate a (ProbePlan, TargetScope) pair for Property 21.

    Produces a self-consistent bundle for exercising
    :meth:`~acdp.agents.red.RedTeamAgent.execute_probe`:

    * ``plan`` — a :class:`~acdp.agents.red.ProbePlan` with an arbitrary list
      of weaknesses (possibly empty, possibly many), a non-empty
      ``target_asset``, and populated ``context_refs``; and
    * ``scope`` — a :class:`~acdp.models.TargetScope` that is always active
      (not expired, not revoked) and contains the plan's ``target_asset``, so
      the authorization check grants the probe and the one-to-one weakness →
      Finding mapping is the only observable outcome.

    The generator spans 0 weaknesses (empty list) through several (max 10) so
    the bijection holds for the empty case and for arbitrary N.
    """
    # Draw a target asset from the shared authorization pool so it can be
    # placed in the scope unambiguously.
    target_asset = draw(_authz_asset)

    weaknesses = draw(
        st.lists(_weakness_text, min_size=0, max_size=10, unique=True)
    )
    context_refs = draw(st.lists(_identifier, min_size=0, max_size=4, unique=True))

    from acdp.agents.red_team_agent import ProbePlan

    plan = ProbePlan(
        plan_id=draw(_identifier),
        task_id=draw(_identifier),
        event_id=draw(_identifier),
        target_asset=target_asset,
        context_refs=context_refs,
        weaknesses=weaknesses,
    )

    # Scope always contains the target asset and never expires during the test.
    scope = TargetScope(
        scope_id=draw(_identifier),
        assets=[target_asset],
        created_at=_RED_TEAM_CREATED_AT,
        expires_at=_RED_TEAM_EXPIRES_AT,
        revoked=False,
    )

    return {"plan": plan, "scope": scope}


# --- Orchestrator strategies (Property 24) ---------------------------------
#
# Property 24 asserts that for any SecurityEvent, Orchestrator.handle_event
# produces a TaskPlan where every task is assigned to a known agent and every
# task carries a non-empty target_scope_id equal to the event's scope.
#
# security_events() generates arbitrary SecurityEvents covering all event types
# and non-empty target_scope_ids.  task_plans() generates TaskPlan instances
# consistent with what handle_event produces (for use in downstream property
# tests that receive a plan directly).

# A small pool of scope ids shared between security_events() and task_plans()
# so that scope ids in plans frequently match those available in a scope
# registry — exercising both the routing logic and the scope-carrying contract.
_scope_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=20,
)

# Payload dictionaries: kept small and string-valued so the SecurityEvent is
# well-formed and fast to construct across many examples.
_event_payload = st.dictionaries(
    keys=_text.filter(lambda s: s != ""),
    values=_text,
    max_size=4,
)


@st.composite
def security_events(draw: st.DrawFn) -> SecurityEvent:
    """Generate :class:`~acdp.models.SecurityEvent` instances for Property 24.

    Every generated event spans all :class:`~acdp.models.SecurityEventType`
    values (``TELEMETRY``, ``PROBE_REQUEST``, ``VULNERABILITY_FINDING``,
    ``PROMPT``, ``UNKNOWN``) with a non-empty ``target_scope_id`` and an
    arbitrary payload. The generator is intentionally broad so the routing
    property (Req 1.1) is exercised across the entire event-type space, not
    just a single type.
    """
    return SecurityEvent(
        event_id=draw(_identifier),
        event_type=draw(st.sampled_from(list(SecurityEventType))),
        target_scope_id=draw(_scope_id),
        payload=draw(_event_payload),
        timestamp=draw(_authz_time),
    )


@st.composite
def task_plans(draw: st.DrawFn) -> TaskPlan:
    """Generate :class:`~acdp.models.TaskPlan` instances for orchestrator tests.

    Produces a self-consistent plan containing one or more tasks, each assigned
    to a known agent identifier (one of ``guardrail``, ``blue_team``,
    ``red_team``, ``devsecops``) and each carrying a non-empty
    ``target_scope_id``. The plan's ``event_id`` is shared across all tasks,
    matching the contract that ``handle_event`` produces (Req 1.2).

    Plans have 1 to 4 tasks with unique agent assignments (no duplicates) so
    the generated plans mirror the de-duplicated output of ``handle_event``.
    """
    from acdp.orchestrator import _KNOWN_AGENTS

    event_id = draw(_identifier)
    scope_id = draw(_scope_id)

    known_agents = list(_KNOWN_AGENTS)
    n_tasks = draw(st.integers(min_value=1, max_value=len(known_agents)))
    agent_ids = draw(
        st.lists(
            st.sampled_from(known_agents),
            min_size=n_tasks,
            max_size=n_tasks,
            unique=True,
        )
    )

    tasks = [
        Task(
            task_id=draw(_identifier),
            event_id=event_id,
            assigned_agent=agent_id,
            target_scope_id=scope_id,
            state=TaskState.PENDING,
        )
        for agent_id in agent_ids
    ]

    return TaskPlan(
        plan_id=draw(_identifier),
        event_id=event_id,
        tasks=tasks,
    )
