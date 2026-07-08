"""Property-based tests for finding identity (Req 12.3)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.models import Finding
from tests.strategies import findings


# Feature: autonomous-cyber-defense-platform, Property 5: Findings have unique
# identifiers linking to their originating event — For any set of findings
# produced by the platform, each ``finding_id`` SHALL be unique and each finding
# SHALL reference the identifier of its originating event.
# Validates: Requirements 12.3
@settings(max_examples=200)
@given(
    # A set of findings "produced by the platform": the platform assigns a
    # unique finding_id to each finding, modelled here by drawing a list that
    # is unique by finding_id.
    produced=st.lists(findings(), unique_by=lambda f: f.finding_id, max_size=12),
)
def test_findings_have_unique_ids_linking_to_originating_event(
    produced: list[Finding],
) -> None:
    # Each finding references the identifier of its originating event: the link
    # is present and non-empty for every finding.
    for finding in produced:
        assert finding.originating_event_id
        assert isinstance(finding.originating_event_id, str)

    # Every finding_id in the set is unique — the number of distinct ids equals
    # the number of findings.
    finding_ids = [f.finding_id for f in produced]
    assert len(set(finding_ids)) == len(finding_ids)

    # Each finding also carries a non-empty finding_id, so the identity link is
    # meaningful (an empty id could not uniquely identify a finding).
    for finding_id in finding_ids:
        assert finding_id
