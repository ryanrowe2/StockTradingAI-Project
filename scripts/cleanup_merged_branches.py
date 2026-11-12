"""Conservatively remove superseded, fully merged temporary GitHub branches.

Only branches with a matching merged PR, or demonstrably contained in the
protected Milestone3 history, qualify. Preserve main and named milestones.
A closed-unmerged PR #5 is an explicitly reviewed superseded exception.
This script defaults to preview mode and requires --execute for deletion.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import time

PRODUCTION = "Milestone3"
SUPERSEDED = {
    "fix/audit-20261111-session-bootstrap-integrity": "fc7d0c80c0145f73e70ee6a563896ceaff5c50de",
}


def request(path: str, *, method: str = "GET") -> object:
    token = os.environ["GITHUB_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repository}/{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "stock-trading-repository-maintenance",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        if method == "DELETE":
            return {}
        return json.loads(response.read().decode("utf-8"))


TRANSIENT_HTTP = {429, 500, 502, 503, 504}


def delete_verified_branch(name: str, expected_sha: str) -> None:
    """Retry transient GitHub gateway failures without deleting a moved ref."""
    ref = urllib.parse.quote(name, safe="")
    for attempt in range(5):
        try:
            latest = request("git/ref/heads/" + ref)
        except urllib.error.HTTPError as error:
            if error.code == 404 and attempt:
                # A prior DELETE may have committed before its gateway failed.
                return
            if error.code not in TRANSIENT_HTTP or attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))
            continue
        if latest["object"]["sha"] != expected_sha:
            raise RuntimeError(f"Unsafe branch-head change: {name}")
        try:
            request("git/refs/heads/" + ref, method="DELETE")
            return
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT_HTTP or attempt == 4:
                raise
            print(
                f"Transient GitHub HTTP {error.code} deleting {name}; "
                f"rechecking SHA before retry {attempt + 2}/5",
                flush=True,
            )
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Exhausted safe deletion attempts: {name}")


def full_collection(path: str) -> list[dict]:
    output: list[dict] = []
    page = 1
    while True:
        separator = "&" if "?" in path else "?"
        result = request(f"{path}{separator}per_page=100&page={page}")
        if not isinstance(result, list):
            raise ValueError(f"Expected list for {path}")
        output.extend(result)
        if len(result) < 100:
            return output
        page += 1


def eligible(branch: dict, merged: dict[str, str]) -> str | None:
    name = branch["name"]
    sha = branch["commit"]["sha"]
    if branch.get("protected") or name in {PRODUCTION, "main", "Milestone2"}:
        return None
    if name in SUPERSEDED and SUPERSEDED[name] == sha:
        return "closed PR #5 explicitly superseded by merged PR #4"
    if not (name.startswith("fix/") or name.startswith("codespace-")):
        return None
    if merged.get(name) == sha:
        return "matching merged pull request"
    compare = request(
        "compare/"
        + urllib.parse.quote(name, safe="")
        + "..."
        + urllib.parse.quote(PRODUCTION, safe="")
    )
    if compare.get("behind_by") == 0 and compare.get("status") in {
        "ahead",
        "identical",
    }:
        return "branch history already contained in Milestone3"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    branches = full_collection("branches")
    closed = full_collection("pulls?state=closed")
    merged = {
        item["head"]["ref"]: item["head"]["sha"]
        for item in closed
        if item.get("merged_at")
        and item["base"]["ref"] == PRODUCTION
        and item["head"]["repo"]
        and item["head"]["repo"]["full_name"] == os.environ["GITHUB_REPOSITORY"]
    }
    deleted: list[str] = []
    preserved: list[str] = []
    for branch in branches:
        name = branch["name"]
        reason = eligible(branch, merged)
        if not reason:
            preserved.append(name)
            continue
        verb = "DELETE" if args.execute else "WOULD DELETE"
        print(f"{verb}: {name} — {reason}", flush=True)
        if args.execute:
            delete_verified_branch(name, branch["commit"]["sha"])
            deleted.append(name)
    if args.execute:
        remaining = {item["name"] for item in full_collection("branches")}
        if set(deleted) & remaining:
            raise RuntimeError(
                f"Deleted branches still exist: {set(deleted) & remaining}"
            )
    print(f"Deleted: {len(deleted)}; preserved: {len(preserved)}", flush=True)
    print("Preserved: " + ", ".join(preserved), flush=True)


if __name__ == "__main__":
    main()
