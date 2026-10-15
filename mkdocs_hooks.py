"""mkdocs hooks: expand the single-source `--8<--` includes and repoint their links.

Four site pages are one-line includes of a file that lives outside `docs/`:
`README.md`, `CHANGELOG.md`, `examples/README.md` and `examples/ffg/README.md`.
Each is browsed on GitHub too, where the directory shape is the repository's,
not the site's. One relative string cannot serve both.

So the sources keep the links a reader of the repo expects, and this hook
repoints them on the way into a page:

| written in `examples/README.md` | rendered on the site                        |
| ------------------------------- | ------------------------------------------- |
| `../docs/assets/x.gif`          | `assets/x.gif`, served by the site itself   |
| `ffg/README.md`                 | `examples-ffg.md`, the page that includes it |
| `bacillus_uncertain_food.py`    | a `blob/main` URL. Source is not published  |
| `../docs/assets/`               | a `tree/main` URL. A directory is not a page |
| `#a-heading`                    | untouched. Anchors already work in both     |

A URL into this repository at any ref, raw or browsed, is read as the path it names
and repointed the same way. The README pins its figures to a commit so its PyPI page
keeps them, and the site still serves its own copy.

A relative link with no target in the repository logs a mkdocs warning, so
`mkdocs build --strict` fails on it. Reference definitions (`[label]: path`)
are repointed and checked alongside inline links. Fenced blocks and inline
code spans are skipped, indented or not.

**Known gap: a four-space indented code block is not recognised**, so a link
inside one is rewritten and dead-link checked as though it were prose. The gap
is deliberate. Any pattern that matches an indented code block also matches
four-space list continuation, and a list item holding a link is ordinary
markdown. Closing this would trade a build that aborts loudly for a link that
ships silently wrong. Use a fence.
"""

from __future__ import annotations

import logging
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BRANCH = "main"
"""Branch the GitHub fallback URLs point at."""

log = logging.getLogger("mkdocs.plugins.single_source")

_INCLUDE = re.compile(r'^--8<--[ \t]+"([^"]+)"[ \t]*$', re.MULTILINE)

# Code to skip and links to repoint, in one pattern. Alternatives are tried left to
# right, so a code region matches first and is handed back untouched and a link inside
# a code sample is never rewritten. Order is the mechanism, not a tidiness choice.
_LINK_SCAN = re.compile(
    # A fenced block. Both rows of ticks may be indented. A fence inside a numbered
    # step is ordinary markdown, and anchoring at column 0 made its links rewritable.
    r"(?P<fence>^[ \t]*(?P<ticks>```+|~~~+)[^\n]*\n.*?^[ \t]*(?P=ticks)[ \t]*$)"
    # An inline span, one line, no backtick inside it. The `+` on the inner class is
    # load-bearing: with `*` a bare ``` line matches here, and when the scan reaches a
    # fence mid-line the fence alternative never gets its turn.
    r"|(?P<code>`+[^`\n]+`+)"
    # A reference definition, `[label]: target "title"`. It resolves to a real link, so
    # it is repointed rather than skipped.
    r"|(?P<ref>^(?P<ref_indent>[ \t]{0,3})\[(?P<label>[^\]\n]+)\]:[ \t]*"
    r"<?(?P<ref_target>[^()<>\s]+)>?(?P<ref_title>[ \t]+\"[^\"\n]*\")?[ \t]*$)"
    # An inline link or image target.
    r"|\]\([ \t]*<?(?P<target>[^()<>\s]+)>?(?P<title>[ \t]+\"[^\"]*\")?[ \t]*\)",
    re.MULTILINE | re.DOTALL,
)


class IncludeError(Exception):
    """An `--8<--` include names a file that is not in the repository."""


@dataclass(frozen=True)
class SiteLayout:
    """Where the repository's files end up once the site is built.

    Attributes:
        repo_root: the repository root, i.e. the directory holding `mkdocs.yml`.
        repo_url: the project's GitHub URL, from the mkdocs `repo_url` setting.
        docs_dir: the docs directory, relative to `repo_root` (here, `docs`).
        includes: each included source file mapped to the page including it.
    """

    repo_root: Path
    repo_url: str
    docs_dir: str
    includes: Mapping[str, str]


def discover_includes(docs_dir: Path) -> dict[str, str]:
    """Find every `--8<--` include in the docs tree.

    Args:
        docs_dir: the mkdocs docs directory.

    Returns:
        Each included file, repo-relative as written in the directive, mapped to
        the docs-relative path of the page that includes it.
    """
    includes: dict[str, str] = {}
    for page in sorted(docs_dir.rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        for source in _INCLUDE.findall(text):
            includes[source] = page.relative_to(docs_dir).as_posix()
    return includes


def resolve_link(target: str, *, source: str, page: str, layout: SiteLayout) -> str:
    """Rewrite one link so it resolves on the site instead of in the repository.

    Args:
        target: the link exactly as written in the source file.
        source: repo-relative path of the file the link is written in.
        page: docs-relative path of the page the text is pasted into.
        layout: where the repository's files land in the built site.

    Returns:
        The rewritten link, `#fragment` preserved. External links, in-page
        anchors and paths that escape the repository come back unchanged.
    """
    repo_path = _repo_path(target, source, layout.repo_url)
    if repo_path is None:
        return target
    _, separator, fragment = target.partition("#")
    return _site_target(repo_path, page=page, layout=layout) + separator + fragment


def rewrite_links(
    text: str, *, source: str, page: str, layout: SiteLayout
) -> tuple[str, list[str]]:
    """Repoint every relative link in one included file, and report dead ones.

    Args:
        text: the included file's markdown.
        source: repo-relative path of that file.
        page: docs-relative path of the page it is pasted into.
        layout: where the repository's files land in the built site.

    Returns:
        The rewritten markdown, and every link whose target does not exist in
        the repository, in the order they appear. Fenced code and inline code
        spans are left alone.
    """
    missing: list[str] = []

    def repointed(target: str) -> str | None:
        """The site target, or None if this link is not ours to rewrite."""
        repo_path = _repo_path(target, source, layout.repo_url)
        if repo_path is None:
            return None
        if not (layout.repo_root / repo_path).exists():
            missing.append(target)
        return resolve_link(target, source=source, page=page, layout=layout)

    def repoint(match: re.Match[str]) -> str:
        # Anything that matched without naming a target is a code region: hand it back.
        if (target := match.group("target")) is not None:
            rewritten = repointed(target)
            if rewritten is None:
                return match.group(0)
            return f"]({rewritten}{match.group('title') or ''})"
        if (target := match.group("ref_target")) is not None:
            rewritten = repointed(target)
            if rewritten is None:
                return match.group(0)
            label, title = match.group("label"), match.group("ref_title") or ""
            return f"{match.group('ref_indent')}[{label}]: {rewritten}{title}"
        return match.group(0)

    return _LINK_SCAN.sub(repoint, text), missing


def expand_includes(
    text: str, *, page: str, layout: SiteLayout
) -> tuple[str, list[tuple[str, str]]]:
    """Replace every `--8<--` include with the named file, links repointed.

    Args:
        text: the page's markdown, include directives and all.
        page: docs-relative path of that page.
        layout: where the repository's files land in the built site.

    Returns:
        The expanded markdown, and every dead link found in the included files
        as (source file, link) pairs.

    Raises:
        IncludeError: an include names a file that is not in the repository.
    """
    missing: list[tuple[str, str]] = []

    def paste(match: re.Match[str]) -> str:
        source = match.group(1)
        path = layout.repo_root / source
        if not path.is_file():
            raise IncludeError(f"{page}: included file {source!r} does not exist")
        included, dead = rewrite_links(
            path.read_text(encoding="utf-8"), source=source, page=page, layout=layout
        )
        missing.extend((source, link) for link in dead)
        # The directive's own newline terminates the paste, so a file ending in a
        # newline would otherwise gain a blank line the source never had.
        return included.rstrip("\n")

    return _INCLUDE.sub(paste, text), missing


def _repo_path(target: str, source: str, repo_url: str) -> str | None:
    """Resolve a link written in `source` to a path relative to the repo root.

    A URL into this repository's files, raw or browsed and at any ref, resolves to the
    path it names. Returns None when the link is not into the repository: an external
    URL, a site-absolute path, an in-page anchor, or a `../` chain that climbs out of
    the tree.
    """
    path = target.partition("#")[0]
    if (named := _path_in_repository_url(path, repo_url)) is not None:
        return named
    if not path or "://" in path or path.startswith(("/", "mailto:")):
        return None
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    return None if resolved.startswith("..") else resolved


def _path_in_repository_url(url: str, repo_url: str) -> str | None:
    """The repo path a raw, blob or tree URL into this repository names, else None."""
    repository = repo_url.rstrip("/").removeprefix("https://github.com/")
    for prefix in (
        f"https://raw.githubusercontent.com/{repository}/",
        f"https://github.com/{repository}/blob/",
        f"https://github.com/{repository}/tree/",
    ):
        if url.startswith(prefix):
            return url.removeprefix(prefix).partition("/")[2].rstrip("/") or None
    return None


def _site_target(repo_path: str, *, page: str, layout: SiteLayout) -> str:
    """Point one resolved repo path at wherever a site reader can reach it."""
    if repo_path in layout.includes:
        return _relative_to_page(layout.includes[repo_path], page)
    if (layout.repo_root / repo_path).is_dir():
        return f"{layout.repo_url}/tree/{BRANCH}/{repo_path}/"
    prefix = f"{layout.docs_dir}/"
    if repo_path.startswith(prefix):
        return _relative_to_page(repo_path.removeprefix(prefix), page)
    return f"{layout.repo_url}/blob/{BRANCH}/{repo_path}"


def _relative_to_page(site_path: str, page: str) -> str:
    """Express a docs-relative path relative to the page linking to it."""
    return posixpath.relpath(site_path, posixpath.dirname(page) or ".")


_layout: SiteLayout | None = None


def on_config(config: Any) -> None:
    """Record where the site's files come from, once per build (mkdocs hook)."""
    global _layout
    repo_root = Path(config["config_file_path"]).parent
    docs_dir = Path(config["docs_dir"])
    _layout = SiteLayout(
        repo_root=repo_root,
        repo_url=str(config["repo_url"]).rstrip("/"),
        docs_dir=docs_dir.relative_to(repo_root).as_posix(),
        includes=discover_includes(docs_dir),
    )


def on_serve(server: Any, config: Any, builder: Any) -> Any:
    """Watch the included files, not only their stubs (mkdocs hook).

    Without this, `mkdocs serve` never notices an edit to a README, because the
    file lives outside `docs/`.
    """
    if _layout is not None:
        for source in _layout.includes:
            server.watch(str(_layout.repo_root / source))
    return server


def on_page_markdown(markdown: str, *, page: Any, config: Any, files: Any) -> str:
    """Expand this page's includes before mkdocs renders it (mkdocs hook)."""
    if _layout is None:
        return markdown
    expanded, missing = expand_includes(
        markdown, page=page.file.src_uri, layout=_layout
    )
    for source, link in missing:
        log.warning("%s: link %r has no target in the repository", source, link)
    return expanded
