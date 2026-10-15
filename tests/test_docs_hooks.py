"""The mkdocs hooks that paste the single-source READMEs into the site."""

from pathlib import Path

import pytest
from mkdocs_hooks import (
    IncludeError,
    SiteLayout,
    discover_includes,
    expand_includes,
    resolve_link,
    rewrite_links,
)

REPO_URL = "https://github.com/inferogenesis/cpomdp"
RAW_URL = "https://raw.githubusercontent.com/inferogenesis/cpomdp"
GALLERY = "examples/README.md"
FFG_GALLERY = "examples/ffg/README.md"


@pytest.fixture
def layout(tmp_path: Path) -> SiteLayout:
    """A miniature of this repository: two galleries, one asset, one script."""
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "demo.gif").write_bytes(b"")
    (tmp_path / "docs" / "guides").mkdir()
    (tmp_path / "docs" / "guides" / "step-by-step.md").write_text("guide\n")
    (tmp_path / "examples" / "ffg").mkdir(parents=True)
    (tmp_path / GALLERY).write_text("gallery\n")
    (tmp_path / FFG_GALLERY).write_text("ffg gallery\n")
    (tmp_path / "examples" / "demo.py").write_text("print()\n")
    return SiteLayout(
        repo_root=tmp_path,
        repo_url=REPO_URL,
        docs_dir="docs",
        includes={GALLERY: "examples.md", FFG_GALLERY: "examples-ffg.md"},
    )


def resolve(target: str, layout: SiteLayout, source: str = GALLERY) -> str:
    """Resolve `target` as written in `source`, pasted into that source's page."""
    return resolve_link(
        target, source=source, page=layout.includes[source], layout=layout
    )


class TestResolveLink:
    """One link at a time, from a gallery README into its page."""

    def test_asset_becomes_a_site_relative_path(self, layout):
        assert resolve("../docs/assets/demo.gif", layout) == "assets/demo.gif"

    def test_asset_from_a_nested_readme_lands_in_the_same_place(self, layout):
        target = resolve("../../docs/assets/demo.gif", layout, source=FFG_GALLERY)
        assert target == "assets/demo.gif"

    def test_a_docs_page_stays_a_docs_page(self, layout):
        target = resolve("../docs/guides/step-by-step.md", layout)
        assert target == "guides/step-by-step.md"

    def test_an_included_readme_becomes_the_page_that_includes_it(self, layout):
        assert resolve("ffg/README.md", layout) == "examples-ffg.md"

    def test_the_mapping_runs_both_ways(self, layout):
        assert resolve("../README.md", layout, source=FFG_GALLERY) == "examples.md"

    def test_unpublished_source_falls_back_to_github(self, layout):
        assert resolve("demo.py", layout) == f"{REPO_URL}/blob/main/examples/demo.py"

    def test_a_directory_falls_back_to_the_github_tree(self, layout):
        target = resolve("../docs/assets/", layout)
        assert target == f"{REPO_URL}/tree/main/docs/assets/"

    def test_a_fragment_survives_the_rewrite(self, layout):
        target = resolve("../docs/guides/step-by-step.md#priors", layout)
        assert target == "guides/step-by-step.md#priors"

    @pytest.mark.parametrize(
        "target",
        [
            "#efe-epistemic-collapse",
            "https://example.com/paper.pdf",
            "mailto:nobody@example.com",
            "/absolute/site/path",
            "../../outside-the-repo.md",
        ],
    )
    def test_links_that_are_already_right_are_left_alone(self, target, layout):
        assert resolve(target, layout) == target

    def test_a_pinned_raw_asset_is_served_by_the_site(self, layout):
        # PyPI needs an absolute image URL, pinned so a later rename cannot break it.
        # The site serves its own copy.
        target = resolve(f"{RAW_URL}/0123abc/docs/assets/demo.gif", layout)
        assert target == "assets/demo.gif"

    def test_a_browsed_url_into_the_repository_resolves_like_its_path(self, layout):
        target = resolve(f"{REPO_URL}/blob/v0.5.0/examples/demo.py", layout)
        assert target == f"{REPO_URL}/blob/main/examples/demo.py"

    def test_a_page_in_a_subdirectory_climbs_back_to_the_assets(self, layout):
        target = resolve_link(
            "../docs/assets/demo.gif",
            source=GALLERY,
            page="guides/gallery.md",
            layout=layout,
        )
        assert target == "../assets/demo.gif"


class TestRewriteLinks:
    """Whole-file rewriting, including what it refuses to touch."""

    def test_images_and_links_are_both_repointed(self, layout):
        text = "![a demo](../docs/assets/demo.gif) and [source](demo.py)\n"
        rewritten, missing = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert "![a demo](assets/demo.gif)" in rewritten
        assert f"[source]({REPO_URL}/blob/main/examples/demo.py)" in rewritten
        assert missing == []

    def test_a_pinned_url_to_a_missing_file_is_reported(self, layout):
        target = f"{RAW_URL}/0123abc/docs/assets/gone.gif"
        _, missing = rewrite_links(
            f"![gone]({target})\n", source=GALLERY, page="examples.md", layout=layout
        )
        assert missing == [target]

    def test_a_link_title_is_kept(self, layout):
        text = '[source](demo.py "the demo")\n'
        rewritten, _ = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert (
            rewritten == f'[source]({REPO_URL}/blob/main/examples/demo.py "the demo")\n'
        )

    def test_fenced_code_is_not_a_link(self, layout):
        text = "```markdown\n[source](demo.py)\n```\n[real](demo.py)\n"
        rewritten, _ = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert "```markdown\n[source](demo.py)\n```" in rewritten
        assert f"[real]({REPO_URL}/blob/main/examples/demo.py)" in rewritten

    def test_a_fence_indented_in_a_list_item_is_not_a_link(self, layout):
        # An ordinary numbered step. Anchoring the ticks at column 0 made these links
        # rewritable *and* reported dead, which aborts `mkdocs build --strict` on an
        # edit that is correct markdown.
        text = "1. run it:\n\n    ```bash\n    see [x](nope.py)\n    ```\n"
        rewritten, missing = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert rewritten == text
        assert missing == []

    def test_an_inline_code_span_is_not_a_link(self, layout):
        text = "write `[source](demo.py)` to link it, as in [real](demo.py)\n"
        rewritten, missing = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert "`[source](demo.py)`" in rewritten
        assert f"[real]({REPO_URL}/blob/main/examples/demo.py)" in rewritten
        assert missing == []

    def test_a_reference_definition_is_repointed(self, layout):
        # `[x]: path` resolves to a real anchor, so leaving it alone shipped the
        # repository's layout to the site and skipped the dead-link check entirely.
        text = '[demo]: ../docs/assets/demo.gif "the demo"\n\nsee ![a][demo]\n'
        rewritten, missing = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert rewritten.startswith('[demo]: assets/demo.gif "the demo"\n')
        assert missing == []

    def test_a_dead_reference_definition_is_reported(self, layout):
        text = "[gone]: ../docs/assets/missing.gif\n"
        _, missing = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert missing == ["../docs/assets/missing.gif"]

    def test_a_scheme_autolink_is_left_alone(self, layout):
        # `<docs/index.md>` is not a link in the first place. CommonMark autolinks
        # need a scheme, and a scheme-bearing one is external. Nothing to rewrite.
        text = "<https://example.com/a.md> and <demo.py>\n"
        rewritten, missing = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert rewritten == text
        assert missing == []

    def test_a_four_space_code_block_is_a_known_gap(self, layout):
        """Pinned, not desired. See the gap named in `mkdocs_hooks`'s docstring.

        The same pattern would match four-space list continuation, so closing this
        would stop rewriting links inside indented list items, putting a silent wrong
        link where a loud build failure used to be. Fixing it breaks this test.
        """
        text = "text\n\n    see [source](demo.py)\n"
        rewritten, _ = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert f"[source]({REPO_URL}/blob/main/examples/demo.py)" in rewritten

    def test_a_link_to_nothing_is_reported(self, layout):
        text = "![gone](../docs/assets/missing.gif)\n"
        _, missing = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert missing == ["../docs/assets/missing.gif"]

    def test_a_fragment_survives_the_whole_file_rewrite(self, layout):
        # The site build goes through here, not through ``resolve_link`` directly, so
        # the fragment has to be pinned on this path too. No gallery link carries one
        # today, which is what makes it worth asserting rather than assuming.
        text = "[priors](../docs/guides/step-by-step.md#priors)\n"
        rewritten, missing = rewrite_links(
            text, source=GALLERY, page="examples.md", layout=layout
        )
        assert rewritten == "[priors](guides/step-by-step.md#priors)\n"
        assert missing == []


class TestExpandIncludes:
    """The directive itself, and the include map behind it."""

    def test_the_directive_is_replaced_by_the_rewritten_file(self, layout):
        (layout.repo_root / GALLERY).write_text("![a](../docs/assets/demo.gif)\n")
        expanded, missing = expand_includes(
            f'--8<-- "{GALLERY}"\n', page="examples.md", layout=layout
        )
        assert expanded == "![a](assets/demo.gif)\n"
        assert missing == []

    def test_dead_links_are_reported_against_their_source_file(self, layout):
        (layout.repo_root / GALLERY).write_text("![a](../docs/assets/gone.gif)\n")
        _, missing = expand_includes(
            f'--8<-- "{GALLERY}"\n', page="examples.md", layout=layout
        )
        assert missing == [(GALLERY, "../docs/assets/gone.gif")]

    def test_a_page_without_a_directive_is_untouched(self, layout):
        text = "# A normal page\n\n[a guide](guides/step-by-step.md)\n"
        assert expand_includes(text, page="index.md", layout=layout) == (text, [])

    def test_including_a_file_that_is_not_there_fails_the_build(self, layout):
        with pytest.raises(IncludeError, match="does not exist"):
            expand_includes('--8<-- "nope.md"\n', page="examples.md", layout=layout)

    def test_the_include_map_is_read_off_the_docs_tree(self, tmp_path):
        docs = tmp_path / "docs"
        (docs / "guides").mkdir(parents=True)
        (docs / "examples.md").write_text(f'--8<-- "{GALLERY}"\n')
        (docs / "guides" / "start.md").write_text('--8<-- "GUIDE.md"\n')
        (docs / "plain.md").write_text("no include here\n")
        assert discover_includes(docs) == {
            GALLERY: "examples.md",
            "GUIDE.md": "guides/start.md",
        }


class TestThisRepository:
    """The real files, so a dead link in a gallery fails here and not on deploy."""

    def test_every_relative_link_in_the_galleries_has_a_target(self):
        repo_root = Path(__file__).resolve().parent.parent
        layout = SiteLayout(
            repo_root=repo_root,
            repo_url=REPO_URL,
            docs_dir="docs",
            includes=discover_includes(repo_root / "docs"),
        )
        dead = {}
        for source, page in layout.includes.items():
            _, missing = rewrite_links(
                (repo_root / source).read_text(encoding="utf-8"),
                source=source,
                page=page,
                layout=layout,
            )
            if missing:
                dead[source] = missing
        assert dead == {}
