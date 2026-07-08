"""GitHub PR Client — concrete PullRequestClient using the GitHub REST API.

The :class:`GitHubPRClient` reads its personal access token exclusively from the
``GIT_TOKEN`` environment variable (never from ``config.yaml`` or constructor
arguments). Any non-2xx response raises an exception that includes the HTTP
status code. Transient 5xx errors are retried with exponential back-off up to
``max_retries`` times.

Usage::

    import os
    os.environ["GIT_TOKEN"] = "ghp_..."
    client = GitHubPRClient(config)
    pr = client.create_pr("org/repo", "Title", "Body", "fix/branch", patch)
"""

from __future__ import annotations

import base64
import math
import os
import time
import uuid

import httpx

from acdp.agents.devsecops_agent import PullRequest
from acdp.connectors.config import GitHubPRClientConfig
from acdp.exceptions import ConfigError

__all__ = ["GitHubPRClient"]

# Version string for the User-Agent header.
_VERSION = "0.1.0"


def _user_agent() -> str:
    return f"acdp-devsecops/{_VERSION}"


class GitHubAPIError(Exception):
    """Raised when the GitHub API returns a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"GitHub API error {status_code}: {body}")


class GitHubPRClient:
    """A :class:`~acdp.agents.devsecops_agent.PullRequestClient` backed by the
    GitHub REST API v3.

    Credentials are read exclusively from the ``GIT_TOKEN`` environment variable
    at construction time. Raises :class:`~acdp.exceptions.ConfigError` if the
    variable is absent or empty.

    Every HTTP request sets ``User-Agent: acdp-devsecops/<version>`` and uses
    exponential back-off retry (base 1 s, factor 2) for 5xx responses, up to
    ``config.max_retries`` retries before giving up.
    """

    def __init__(self, config: GitHubPRClientConfig | None = None) -> None:
        token = os.environ.get("GIT_TOKEN", "").strip()
        if not token:
            raise ConfigError("GIT_TOKEN")

        self._token = token
        cfg = config or GitHubPRClientConfig()
        self._base_url = cfg.github_api_base_url.rstrip("/")
        self._max_retries = cfg.max_retries

        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": _user_agent(),
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # PullRequestClient Protocol
    # ------------------------------------------------------------------

    def create_pr(
        self,
        repo: str,
        title: str,
        body: str,
        branch: str,
        patch: str,
    ) -> PullRequest:
        """Open a review-required pull request on ``repo``.

        Steps:
        1. Resolve the repository's ``default_branch``.
        2. Get the SHA of the tip of ``default_branch``.
        3. Create a new branch (appending a UUID suffix on 409 Conflict).
        4. Commit ``patch`` as a base64-encoded file on the new branch.
        5. Open the PR and return a :class:`~acdp.agents.devsecops_agent.PullRequest`
           with ``requires_review=True``.
        """
        # Step 1: resolve default branch
        repo_data = self._get(f"/repos/{repo}")
        default_branch: str = repo_data["default_branch"]

        # Step 2: get SHA of default branch tip
        ref_data = self._get(f"/repos/{repo}/git/ref/heads/{default_branch}")
        sha: str = ref_data["object"]["sha"]

        # Step 3: create branch (handle 409 by appending UUID suffix)
        actual_branch = branch
        try:
            self._post(
                f"/repos/{repo}/git/refs",
                json={
                    "ref": f"refs/heads/{actual_branch}",
                    "sha": sha,
                },
            )
        except GitHubAPIError as exc:
            if exc.status_code == 409:
                # Branch already exists — append a unique suffix (Req 5.9)
                actual_branch = f"{branch}-{uuid.uuid4().hex[:8]}"
                self._post(
                    f"/repos/{repo}/git/refs",
                    json={
                        "ref": f"refs/heads/{actual_branch}",
                        "sha": sha,
                    },
                )
            else:
                raise

        # Step 4: commit patch as base64-encoded file content
        patch_filename = "security-patch.diff"
        encoded_content = base64.b64encode(patch.encode("utf-8")).decode("ascii")

        # Check if file already exists to get its SHA (required by GitHub API for updates)
        file_sha: str | None = None
        try:
            existing = self._get(f"/repos/{repo}/contents/{patch_filename}")
            file_sha = existing.get("sha")
        except GitHubAPIError as exc:
            if exc.status_code != 404:
                raise

        put_body: dict = {
            "message": f"Security fix patch: {title}",
            "content": encoded_content,
            "branch": actual_branch,
        }
        if file_sha is not None:
            put_body["sha"] = file_sha

        self._put(f"/repos/{repo}/contents/{patch_filename}", json=put_body)

        # Step 5: open pull request
        pr_data = self._post(
            f"/repos/{repo}/pulls",
            json={
                "title": title,
                "body": body,
                "head": actual_branch,
                "base": default_branch,
            },
        )

        return PullRequest(
            pr_id=str(pr_data.get("number", uuid.uuid4())),
            repo=repo,
            title=title,
            body=body,
            patch=patch,
            requires_review=True,
        )

    # ------------------------------------------------------------------
    # Internal HTTP helpers with retry
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, *, json: dict) -> dict:
        return self._request("POST", path, json=json)

    def _put(self, path: str, *, json: dict) -> dict:
        return self._request("PUT", path, json=json)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Execute an HTTP request with exponential back-off retry for 5xx errors."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.RequestError as exc:
                # Network errors are non-transient from retry perspective —
                # we raise immediately per the design spec which says 4xx and
                # ConfigErrors raise immediately; network errors also raise immediately.
                raise

            if response.is_success:
                # Return empty dict for 204 No Content
                if response.status_code == 204 or not response.content:
                    return {}
                return response.json()

            # Non-2xx response
            status_code = response.status_code
            body = response.text

            if status_code >= 500:
                # Transient error — retry with exponential back-off
                last_exc = GitHubAPIError(status_code, body)
                if attempt < self._max_retries:
                    delay = math.pow(2, attempt)  # 1s, 2s, 4s ...
                    time.sleep(delay)
                    continue
                # Max retries exhausted
                raise GitHubAPIError(status_code, body)
            else:
                # 4xx or other non-5xx — raise immediately (non-transient)
                raise GitHubAPIError(status_code, body)

        # Should never reach here
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Unexpected exit from retry loop for {method} {url}")
