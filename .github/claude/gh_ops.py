#!/usr/bin/env python3
"""Guarded GitHub operations for the weekly Claude upkeep run.

The upkeep workflow copies this file and `labels.json` into a read-only directory under
`$RUNNER_TEMP` before Claude starts. That copy is the only command Claude may run, so
every write it can make to GitHub passes these checks:

- labels are added, never removed, and only those marked `"claude": true`
- issues it files are capped per run and by backlog, and always carry
  `claude-proposed`
- pull requests are fixes, docs, tests or chores. Features stay with contributors.
  Each one comes from a fresh `claude/upkeep-*` branch, stays out of the forbidden
  paths and is capped in size and in number
- the upkeep log gets one comment per run

Claude also reads git through `changes` and `history`, because it has no shell.

The caps come from the workflow step's environment, which Claude cannot set. The
per-run state lives at `CLAUDE_OPS_STATE`. The workflow denies Claude's file tools
everything under `$RUNNER_TEMP`, so neither can be edited from inside the run.

`sync-labels` is for a person or the label workflow. It creates or updates every
label in `labels.json` and deletes nothing.

Standard library and the `gh` CLI only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

DEFAULT_LIMITS: dict[str, Any] = {
    "triage_per_run": 15,
    "propose_per_run": 2,
    "propose_backlog": 6,
    "pr_per_run": 1,
    "pr_backlog": 2,
    "pr_max_files": 10,
    "pr_max_lines": 400,
    "forbidden_paths": [".github/"],
}

# `feat` is absent on purpose. Features are filed as issues for contributors.
PR_TITLE = re.compile(
    r"^(fix|docs|test|chore|refactor|perf|style|build)(\([a-z0-9_./-]+\))?: \S.{0,100}$"
)
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
BRANCH_PREFIX = "claude/upkeep-"
LOG_LABEL = "upkeep-log"
LOG_TITLE = "Weekly upkeep log"
PROPOSED_LABEL = "claude-proposed"
TRIAGED_LABEL = "triaged"
SCRATCH_DIR = ".claude-out"
BODY_SHOWN = 2000
BODY_MAX = 20000

PROPOSE_FOOTER = (
    "\n\n---\n\n_Filed by the weekly upkeep run. A maintainer has not confirmed the "
    "scope yet. Comment here before you start and they will confirm it or narrow it._"
)
PR_FOOTER = (
    "\n\n---\n\n_Opened by the weekly upkeep run. It never merges its own work. "
    "Review it like any other pull request._"
)


class Refused(Exception):
    """A request the guards turn down. Claude reads the message and carries on."""


def _beside(name: str) -> Path:
    for candidate in (HERE / name, HERE.parent / name):
        if candidate.exists():
            return candidate
    raise SystemExit(f"{name} not found beside {HERE}")


def load_labels() -> list[dict[str, Any]]:
    """The label definitions from labels.json."""
    return json.loads(_beside("labels.json").read_text())["labels"]


def claude_labels() -> set[str]:
    """Names of the labels the upkeep run may add."""
    return {label["name"] for label in load_labels() if label.get("claude")}


def load_limits() -> dict[str, Any]:
    """The caps, with the step's `CLAUDE_*` environment over the defaults."""
    limits = dict(DEFAULT_LIMITS)
    for key, var in (
        ("triage_per_run", "CLAUDE_TRIAGE_CAP"),
        ("propose_per_run", "CLAUDE_PROPOSE_CAP"),
    ):
        if os.environ.get(var):
            limits[key] = int(os.environ[var])
    if os.environ.get("CLAUDE_FORBIDDEN_PATHS"):
        limits["forbidden_paths"] = os.environ["CLAUDE_FORBIDDEN_PATHS"].split()
    return limits


def state_path() -> Path:
    """Where this run's counters live. Required inside Actions."""
    if os.environ.get("CLAUDE_OPS_STATE"):
        return Path(os.environ["CLAUDE_OPS_STATE"])
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise Refused("CLAUDE_OPS_STATE is unset, so no write can be counted")
    return HERE / "state.json"


def load_state() -> dict[str, Any]:
    """What this run has written so far."""
    path = state_path()
    if path.exists():
        return json.loads(path.read_text())
    return {"triaged": [], "proposed": 0, "prs": 0, "logged": False}


def save_state(state: dict[str, Any]) -> None:
    """Record what this run has written. Called before the write it counts."""
    state_path().write_text(json.dumps(state))


def repo() -> str:
    """The `owner/name` this run acts on."""
    return os.environ["GITHUB_REPOSITORY"]


def gh(*args: str, payload: dict[str, Any] | None = None) -> Any:
    """Run `gh api` and return the parsed JSON, or raise Refused with gh's error."""
    cmd = ["gh", "api", *args]
    data = None
    if payload is not None:
        cmd += ["--input", "-"]
        data = json.dumps(payload)
    out = subprocess.run(cmd, input=data, capture_output=True, text=True)
    if out.returncode != 0:
        raise Refused(f"gh api {args[0]} failed: {out.stderr.strip()[:500]}")
    return json.loads(out.stdout) if out.stdout.strip() else None


def gh_all(path: str) -> list[dict[str, Any]]:
    """Every page of a list endpoint, flattened."""
    pages = gh(path, "--paginate", "--slurp")
    return [item for page in pages for item in page]


# Global and system git config are ignored, so a planted `~/.gitconfig` can't add hooks,
# an fsmonitor or an external diff. The checkout's own config holds the identity and
# remote the action set up.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


def git(*args: str) -> str:
    """Run git and return stdout, or raise Refused with git's error."""
    out = subprocess.run(["git", *args], capture_output=True, text=True, env=GIT_ENV)
    if out.returncode != 0:
        raise Refused(f"git {args[0]} failed: {out.stderr.strip()[:500]}")
    return out.stdout


def default_branch() -> str:
    """The branch upkeep pull requests target."""
    return (
        os.environ.get("CLAUDE_BASE_BRANCH") or gh(f"repos/{repo()}")["default_branch"]
    )


def label_names(item: dict[str, Any]) -> list[str]:
    """Label names on an issue or pull request."""
    return [label["name"] for label in item.get("labels", [])]


def open_issues() -> list[dict[str, Any]]:
    """Open issues, excluding pull requests."""
    items = gh_all(f"repos/{repo()}/issues?state=open&per_page=100")
    return [item for item in items if "pull_request" not in item]


# GitHub token shapes, so a body can never carry one to a public page.
TOKEN = re.compile(
    r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|sk-ant-[\w-]{20,})"
    r"|x-access-token:\S+"
)


def read_body(path: str) -> str:
    """Read a body file from the scratch folder, refusing anything else.

    Bodies are posted publicly. Reading only from `.claude-out/` keeps the tool from
    publishing `.git/config` or any other file outside it.
    """
    scratch = (Path.cwd() / SCRATCH_DIR).resolve()
    target = Path(path).resolve()
    if scratch not in target.parents:
        raise Refused(f"body files must live under {SCRATCH_DIR}/")
    body = target.read_text().strip()
    if TOKEN.search(body):
        raise Refused(f"{path} contains something shaped like a GitHub token")
    if not body:
        raise Refused(f"{path} is empty")
    if len(body) > BODY_MAX:
        raise Refused(f"{path} is {len(body)} characters, over the {BODY_MAX} cap")
    return body


# Reads. Their output carries text written by strangers, which is data to Claude.


def cmd_issues(args: argparse.Namespace) -> None:
    """Print open issues as JSON, bodies truncated."""
    items = open_issues()
    if args.untriaged:
        items = [item for item in items if TRIAGED_LABEL not in label_names(item)]
    print(
        json.dumps(
            [
                {
                    "number": item["number"],
                    "title": item["title"],
                    "labels": label_names(item),
                    "author": item["user"]["login"],
                    "created_at": item["created_at"],
                    "comments": item["comments"],
                    "body": (item.get("body") or "")[:BODY_SHOWN],
                }
                for item in items
            ],
            indent=1,
        )
    )


def cmd_prs(args: argparse.Namespace) -> None:
    """Print open pull requests with a tally of their check runs."""
    pulls = gh_all(f"repos/{repo()}/pulls?state=open&per_page=100")
    report = []
    for pull in pulls:
        pages = gh(
            f"repos/{repo()}/commits/{pull['head']['sha']}/check-runs?per_page=100",
            "--paginate",
            "--slurp",
        )
        tally: dict[str, int] = {}
        for run in (run for page in pages for run in page["check_runs"]):
            key = run["conclusion"] or run["status"]
            tally[key] = tally.get(key, 0) + 1
        report.append(
            {
                "number": pull["number"],
                "title": pull["title"],
                "author": pull["user"]["login"],
                "branch": pull["head"]["ref"],
                "draft": pull["draft"],
                "labels": label_names(pull),
                "updated_at": pull["updated_at"],
                "checks": tally,
            }
        )
    print(json.dumps(report, indent=1))


def cmd_runs(args: argparse.Namespace) -> None:
    """Print runs on the default branch that did not succeed."""
    data = gh(f"repos/{repo()}/actions/runs?branch={default_branch()}&per_page=30")
    print(
        json.dumps(
            [
                {
                    "id": run["id"],
                    "workflow": run["name"],
                    "conclusion": run["conclusion"],
                    "created_at": run["created_at"],
                    "url": run["html_url"],
                }
                for run in data["workflow_runs"]
                if run["conclusion"] not in ("success", "skipped", "neutral", None)
            ],
            indent=1,
        )
    )


def cmd_failed_log(args: argparse.Namespace) -> None:
    """Print the tail of a run's failed-step log."""
    out = subprocess.run(
        ["gh", "run", "view", str(args.run_id), "--log-failed", "-R", repo()],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise Refused(f"gh run view failed: {out.stderr.strip()[:500]}")
    print("\n".join(out.stdout.splitlines()[-300:]))


def cmd_changes(args: argparse.Namespace) -> None:
    """Print `git status` and the working-tree diff, so Claude can check its edits."""
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = git("diff", "--no-ext-diff", "--no-textconv", "--no-color")
    lines = diff.splitlines()
    print(status + "\n" + "\n".join(lines[:600]))
    if len(lines) > 600:
        print(f"... {len(lines) - 600} more diff lines")


def cmd_history(args: argparse.Namespace) -> None:
    """Print recent commits, optionally for one path, with their file stats."""
    if args.path and args.path.startswith("-"):
        raise Refused("the path cannot start with a dash")
    count = max(1, min(args.count, 50))
    cmd = [
        "log",
        "--no-ext-diff",
        "--no-color",
        f"-n{count}",
        "--date=short",
        "--format=%h %ad %an%n    %s",
        "--stat",
    ]
    if args.path:
        cmd += ["--", args.path]
    print(git(*cmd))


# Writes. Each one is checked against labels.json, the caps and this run's state. The
# state is saved before the write it counts, so a crash part-way fails closed.


def cmd_label(args: argparse.Namespace) -> None:
    """Add allowed labels and `triaged` to one issue."""
    allowed = claude_labels()
    unknown = [name for name in args.labels if name not in allowed]
    if unknown:
        raise Refused(
            f"not labels this run may apply: {unknown}. Allowed: {sorted(allowed)}"
        )
    state = load_state()
    cap = load_limits()["triage_per_run"]
    if args.issue not in state["triaged"] and len(state["triaged"]) >= cap:
        raise Refused(f"this run has labelled {cap} issues, its cap")
    if args.issue not in state["triaged"]:
        state["triaged"].append(args.issue)
    save_state(state)
    gh(
        f"repos/{repo()}/issues/{args.issue}/labels",
        "-X",
        "POST",
        payload={"labels": [*args.labels, TRIAGED_LABEL]},
    )
    print(f"#{args.issue}: added {[*args.labels, TRIAGED_LABEL]}")


def cmd_propose(args: argparse.Namespace) -> None:
    """File one contributor-ready issue."""
    limits = load_limits()
    state = load_state()
    if limits["propose_per_run"] == 0:
        raise Refused("this repository takes no filed issues. Skip step 4.")
    if state["proposed"] >= limits["propose_per_run"]:
        raise Refused(f"this run has filed {limits['propose_per_run']} issues, its cap")
    allowed = claude_labels()
    labels = [name.strip() for name in args.labels.split(",") if name.strip()]
    unknown = [name for name in labels if name not in allowed]
    if unknown:
        raise Refused(f"not labels this run may apply: {unknown}")
    title = args.title.strip()
    if not 10 <= len(title) <= 120:
        raise Refused("the title must be 10 to 120 characters")
    body = read_body(args.body_file)
    issues = open_issues()
    backlog = sum(PROPOSED_LABEL in label_names(item) for item in issues)
    if backlog >= limits["propose_backlog"]:
        raise Refused(
            f"{backlog} claude-proposed issues are still open, at or over the "
            f"backlog cap of {limits['propose_backlog']}. File none this week."
        )
    if any(item["title"].strip().lower() == title.lower() for item in issues):
        raise Refused("an open issue already has this title")
    state["proposed"] += 1
    save_state(state)
    created = gh(
        f"repos/{repo()}/issues",
        "-X",
        "POST",
        payload={
            "title": title,
            "body": body + PROPOSE_FOOTER,
            "labels": sorted({*labels, PROPOSED_LABEL, TRIAGED_LABEL}),
        },
    )
    print(f"filed #{created['number']}: {created['html_url']}")


def _staged_or_refuse(limits: dict[str, Any]) -> list[str]:
    """Stage the working tree, minus the scratch folder, and check it against caps."""
    git("add", "-A", "--", ".", f":(exclude){SCRATCH_DIR}")
    # `--no-renames` lists both ends of a move, so moving a forbidden file is caught.
    paths = [
        p
        for p in git("diff", "--cached", "--no-renames", "--name-only", "-z").split(
            "\0"
        )
        if p
    ]
    if not paths:
        raise Refused("nothing to commit outside the scratch folder")
    problem = None
    forbidden = [
        p
        for p in paths
        if any(p == f.rstrip("/") or p.startswith(f) for f in limits["forbidden_paths"])
    ]
    if forbidden:
        problem = f"touches forbidden paths {forbidden}"
    elif len(paths) > limits["pr_max_files"]:
        problem = f"{len(paths)} files, over the cap of {limits['pr_max_files']}"
    else:
        lines = 0
        for row in git("diff", "--cached", "--no-renames", "--numstat").splitlines():
            added, deleted, _ = row.split("\t", 2)
            if added == "-" or deleted == "-":
                problem = "includes a binary file"
                break
            lines += int(added) + int(deleted)
        if problem is None and lines > limits["pr_max_lines"]:
            problem = f"{lines} changed lines, over the cap of {limits['pr_max_lines']}"
    if problem:
        git("reset", "-q")
        raise Refused(
            f"the change {problem}. Narrow it, or describe it in the log instead."
        )
    return paths


def cmd_open_pr(args: argparse.Namespace) -> None:
    """Commit the working tree to a new upkeep branch and open a PR."""
    limits = load_limits()
    state = load_state()
    if state["prs"] >= limits["pr_per_run"]:
        raise Refused(
            f"this run has opened {limits['pr_per_run']} pull requests, its cap"
        )
    if not PR_TITLE.match(args.title):
        raise Refused(
            "the title must be a Conventional Commit of type fix, docs, test, chore, "
            "refactor, perf, style or build. Features go to contributors as issues."
        )
    if not SLUG.match(args.slug):
        raise Refused(
            "the slug must be lowercase letters, digits and hyphens, at most 41"
        )
    body = read_body(args.body_file)
    base = default_branch()
    if git("rev-parse", "--abbrev-ref", "HEAD").strip() != base:
        raise Refused(f"start from {base}. The previous open-pr returns there itself.")
    pulls = gh_all(f"repos/{repo()}/pulls?state=open&per_page=100")
    backlog = sum(pull["head"]["ref"].startswith(BRANCH_PREFIX) for pull in pulls)
    if backlog >= limits["pr_backlog"]:
        raise Refused(
            f"{backlog} upkeep pull requests are still open, at or over the backlog "
            f"cap of {limits['pr_backlog']}. Open none this week."
        )
    _staged_or_refuse(limits)
    state["prs"] += 1
    save_state(state)
    branch = f"{BRANCH_PREFIX}{dt.date.today():%Y%m%d}-{args.slug}"
    if git("ls-remote", "--heads", "origin", branch).strip():
        git("reset", "-q")
        raise Refused(f"{branch} already exists on the remote")
    git("switch", "-q", "-c", branch)
    try:
        git("commit", "-q", "-m", args.title)
        git("push", "-q", "origin", f"HEAD:refs/heads/{branch}")
        created = gh(
            f"repos/{repo()}/pulls",
            "-X",
            "POST",
            payload={
                "title": args.title,
                "head": branch,
                "base": base,
                "body": body + PR_FOOTER,
            },
        )
    finally:
        subprocess.run(["git", "switch", "-q", base], capture_output=True, env=GIT_ENV)
    print(f"opened #{created['number']}: {created['html_url']}")


def cmd_log(args: argparse.Namespace) -> None:
    """Post this run's summary to the upkeep log issue."""
    state = load_state()
    if state["logged"]:
        raise Refused("this run has already posted its log comment")
    body = read_body(args.body_file)
    state["logged"] = True
    save_state(state)
    logs = gh_all(f"repos/{repo()}/issues?state=open&labels={LOG_LABEL}&per_page=100")
    logs = [item for item in logs if "pull_request" not in item]
    if logs:
        number = min(item["number"] for item in logs)
    else:
        number = gh(
            f"repos/{repo()}/issues",
            "-X",
            "POST",
            payload={
                "title": LOG_TITLE,
                "body": "One comment per weekly upkeep run. Unsubscribe to stop the "
                "notifications. Close it and the next run opens a fresh one.",
                "labels": [LOG_LABEL],
            },
        )["number"]
    comment = gh(
        f"repos/{repo()}/issues/{number}/comments", "-X", "POST", payload={"body": body}
    )
    print(f"logged to #{number}: {comment['html_url']}")


def cmd_sync_labels(args: argparse.Namespace) -> None:
    """Create or update every label in labels.json."""
    existing = {
        label["name"].lower(): label
        for label in gh_all(f"repos/{repo()}/labels?per_page=100")
    }
    for label in load_labels():
        payload = {
            "color": label["color"],
            "description": label.get("description", ""),
        }
        found = existing.get(label["name"].lower())
        if found is None:
            gh(
                f"repos/{repo()}/labels",
                "-X",
                "POST",
                payload={"name": label["name"], **payload},
            )
            print(f"created {label['name']}")
        elif (found["color"], found.get("description") or "") != (
            payload["color"],
            payload["description"],
        ):
            name = urllib.parse.quote(found["name"], safe="")
            gh(f"repos/{repo()}/labels/{name}", "-X", "PATCH", payload=payload)
            print(f"updated {label['name']}")


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and run it. Exit 2 on a refusal."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    issues = sub.add_parser("issues", help="open issues as JSON")
    issues.add_argument(
        "--untriaged", action="store_true", help="only those without `triaged`"
    )
    issues.set_defaults(run=cmd_issues)

    sub.add_parser("prs", help="open pull requests and their checks").set_defaults(
        run=cmd_prs
    )
    sub.add_parser("runs", help="failed runs on the default branch").set_defaults(
        run=cmd_runs
    )

    failed = sub.add_parser("failed-log", help="the failed steps' log of one run")
    failed.add_argument("run_id", type=int)
    failed.set_defaults(run=cmd_failed_log)

    sub.add_parser("changes", help="git status and the working-tree diff").set_defaults(
        run=cmd_changes
    )
    history = sub.add_parser("history", help="recent commits with file stats")
    history.add_argument("--path", default="", help="limit to one file or folder")
    history.add_argument("-n", dest="count", type=int, default=15)
    history.set_defaults(run=cmd_history)

    label = sub.add_parser("label", help="add labels to an issue (and `triaged`)")
    label.add_argument("issue", type=int)
    label.add_argument("labels", nargs="+")
    label.set_defaults(run=cmd_label)

    propose = sub.add_parser("propose", help="file a contributor-ready issue")
    propose.add_argument("--title", required=True)
    propose.add_argument("--labels", required=True, help="comma-separated")
    propose.add_argument("--body-file", required=True)
    propose.set_defaults(run=cmd_propose)

    pull = sub.add_parser(
        "open-pr", help="commit the working tree to a new branch and open a PR"
    )
    pull.add_argument("--slug", required=True)
    pull.add_argument("--title", required=True)
    pull.add_argument("--body-file", required=True)
    pull.set_defaults(run=cmd_open_pr)

    log = sub.add_parser("log", help="post this run's summary to the upkeep log issue")
    log.add_argument("--body-file", required=True)
    log.set_defaults(run=cmd_log)

    sub.add_parser(
        "sync-labels", help="create or update labels.json's labels"
    ).set_defaults(run=cmd_sync_labels)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except Refused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
