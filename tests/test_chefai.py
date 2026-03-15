"""Tests for chefai.

Coverage targets:
- chefai.__version__
- extractor.sanitize_filename
- extractor.parse_preparazioni_md  (happy path, failure modes, edge cases)
- extractor.export_recipes_to_markdown  (round-trip, duplicate titles)
"""

from pathlib import Path

import pytest

import chefai
from chefai.extractor import (
    export_recipes_to_markdown,
    parse_preparazioni_md,
    sanitize_filename,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

#: Path to the real processed file used in production.
PREPARAZIONI_PATH = Path("data/processed/preparazioni-di-base.md")


def _minimal_md(text_block: str, titolo: str = "TEST SECTION") -> str:
    """Build a minimal preparazioni-style markdown string for unit tests."""
    return (
        f"---\ntitolo: {titolo}\n---\n\n"
        "```text\n"
        f"{text_block}\n"
        "```\n"
    )


# ---------------------------------------------------------------------------
# chefai package
# ---------------------------------------------------------------------------


def test_version():
    """Package exposes a non-empty __version__ string."""
    assert isinstance(chefai.__version__, str)
    assert len(chefai.__version__) > 0


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


def test_sanitize_plain():
    """ASCII titles are lowercased and space-collapsed."""
    assert sanitize_filename("Besciamella") == "besciamella"
    assert sanitize_filename("Brodo Di Carne") == "brodo-di-carne"


def test_sanitize_accents():
    """Accented characters are stripped to their ASCII base."""
    assert sanitize_filename("Crêpe") == "crepe"
    assert sanitize_filename("Ragù") == "ragu"


def test_sanitize_empty_and_symbols_only():
    """Empty or symbol-only titles fall back to the sentinel."""
    assert sanitize_filename("") == "ricetta-senza-titolo"
    assert sanitize_filename("!!!###") == "ricetta-senza-titolo"


def test_sanitize_truncation():
    """Titles longer than max_length are truncated without a trailing dash."""
    result = sanitize_filename("a" * 100, max_length=80)
    assert len(result) <= 80
    assert not result.endswith("-")


# ---------------------------------------------------------------------------
# parse_preparazioni_md — failure modes
# ---------------------------------------------------------------------------


def test_parse_missing_file():
    """FileNotFoundError is raised for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        parse_preparazioni_md("this_file_does_not_exist.md")


def test_parse_no_text_block(tmp_path: Path):
    """ValueError is raised when the file has no ```text block."""
    f = tmp_path / "no_block.md"
    f.write_text("---\ntitolo: EMPTY\n---\n# Nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"No [`]+text block"):
        parse_preparazioni_md(f)


# ---------------------------------------------------------------------------
# parse_preparazioni_md — unit tests with synthetic input
# ---------------------------------------------------------------------------


def test_parse_single_recipe(tmp_path: Path):
    """A single well-formed recipe is parsed correctly."""
    md = _minimal_md(
        "--- pagina 1 ---\n"
        "BESCIAMELLA\n"
        "INGREDIENTI\n"
        "40 G DI BURRO,\n"
        "50 G DI FARINA\n"
        "Fate fondere il burro e unite la farina.\n"
    )
    f = tmp_path / "test.md"
    f.write_text(md, encoding="utf-8")

    recipes = parse_preparazioni_md(f)

    assert len(recipes) == 1
    r = recipes[0]
    assert r["title"] == "Besciamella"
    assert r["ingredients"] == ["40 G DI BURRO", "50 G DI FARINA"]
    assert len(r["instructions"]) == 1
    assert "burro" in r["instructions"][0].lower()
    assert r["pages"] == [1]


def test_parse_bare_allcaps_ingredient_not_treated_as_title(tmp_path: Path):
    """
    Regression: bare ALL-CAPS words like SALE that appear as the final
    ingredient (no unit, no comma) must NOT be parsed as a new recipe title.
    """
    md = _minimal_md(
        "--- pagina 1 ---\n"
        "BRODO DI CARNE\n"
        "INGREDIENTI\n"
        "600 G DI MANZO,\n"
        "1 CIPOLLA,\n"
        "SALE\n"
        "Cuocete per 4 ore a fuoco lento.\n"
    )
    f = tmp_path / "test.md"
    f.write_text(md, encoding="utf-8")

    recipes = parse_preparazioni_md(f)

    assert len(recipes) == 1, (
        f"Expected 1 recipe, got {len(recipes)}: {[r['title'] for r in recipes]}. "
        "SALE was wrongly identified as a recipe title."
    )
    assert "SALE" in recipes[0]["ingredients"]
    assert recipes[0]["instructions"]  # instructions were NOT eaten by SALE


def test_parse_multipage_recipe(tmp_path: Path):
    """A recipe that spans a page boundary accumulates both page numbers."""
    md = _minimal_md(
        "--- pagina 5 ---\n"
        "BURRO CHIARIFICATO\n"
        "INGREDIENTI\n"
        "250 G DI BURRO\n"
        "--- pagina 6 ---\n"
        "Fate fondere il burro a bagnomaria per 30 minuti.\n"
    )
    f = tmp_path / "test.md"
    f.write_text(md, encoding="utf-8")

    recipes = parse_preparazioni_md(f)

    assert len(recipes) == 1
    assert recipes[0]["pages"] == [5, 6]


def test_parse_section_heading_excluded(tmp_path: Path):
    """The document-level section heading must not appear as a recipe."""
    md = _minimal_md(
        "--- pagina 1 ---\n"
        "TEST SECTION\n"           # matches titolo — should be skipped
        "--- pagina 2 ---\n"
        "BESCIAMELLA\n"
        "INGREDIENTI\n"
        "40 G DI BURRO\n"
        "Fate fondere.\n",
        titolo="TEST SECTION",
    )
    f = tmp_path / "test.md"
    f.write_text(md, encoding="utf-8")

    recipes = parse_preparazioni_md(f)

    titles = [r["title"] for r in recipes]
    assert "Test Section" not in titles
    assert "Besciamella" in titles


def test_parse_instruction_paragraphs_split(tmp_path: Path):
    """Two prose paragraphs (separated by a blank line) become two steps."""
    md = _minimal_md(
        "--- pagina 1 ---\n"
        "POLENTA\n"
        "INGREDIENTI\n"
        "400 G DI FARINA DI MAIS,\n"
        "SALE\n"
        "Portate a ebollizione l'acqua e versate la farina.\n"
        "\n"
        "Continuate a mescolare per un'ora.\n"
    )
    f = tmp_path / "test.md"
    f.write_text(md, encoding="utf-8")

    recipes = parse_preparazioni_md(f)

    assert len(recipes[0]["instructions"]) == 2


# ---------------------------------------------------------------------------
# parse_preparazioni_md — integration test against the real source file
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PREPARAZIONI_PATH.exists(),
    reason="Real source file not present (data/processed/preparazioni-di-base.md)",
)
def test_parse_real_file_recipe_count():
    """Real source file yields exactly 18 recipes."""
    recipes = parse_preparazioni_md(PREPARAZIONI_PATH)
    assert len(recipes) == 18


@pytest.mark.skipif(
    not PREPARAZIONI_PATH.exists(),
    reason="Real source file not present",
)
def test_parse_real_file_all_recipes_have_ingredients_and_instructions():
    """Every recipe parsed from the real file has at least one ingredient and one step."""
    recipes = parse_preparazioni_md(PREPARAZIONI_PATH)
    for r in recipes:
        assert r["ingredients"], f"{r['title']!r} has no ingredients"
        assert r["instructions"], f"{r['title']!r} has no instructions"


@pytest.mark.skipif(
    not PREPARAZIONI_PATH.exists(),
    reason="Real source file not present",
)
def test_parse_real_file_known_recipes():
    """Spot-check specific recipes that have previously caused parsing bugs."""
    recipes = parse_preparazioni_md(PREPARAZIONI_PATH)
    by_title = {r["title"]: r for r in recipes}

    # BRODO DI CARNE: SALE must be an ingredient, not a title
    brodo = by_title["Brodo Di Carne"]
    assert "SALE" in brodo["ingredients"]
    assert brodo["instructions"]

    # BURRO CHIARIFICATO: spans pages 17 and 18
    burro = by_title["Burro Chiarificato"]
    assert 17 in burro["pages"]
    assert 18 in burro["pages"]


# ---------------------------------------------------------------------------
# export_recipes_to_markdown
# ---------------------------------------------------------------------------


def test_export_creates_files(tmp_path: Path):
    """Each recipe dict produces one markdown file in the output directory."""
    recipes = [
        {"title": "Besciamella", "ingredients": ["40 G DI BURRO"], "instructions": ["Fate fondere."]},
        {"title": "Ragù", "ingredients": ["400 G DI MANZO"], "instructions": ["Cuocete."]},
    ]
    export_recipes_to_markdown(recipes, output_dir=tmp_path, include_raw_text=False)

    assert (tmp_path / "besciamella.md").exists()
    assert (tmp_path / "ragu.md").exists()


def test_export_file_contains_yaml_frontmatter(tmp_path: Path):
    """Exported files start with a valid YAML front-matter block."""
    recipes = [{"title": "Besciamella", "ingredients": [], "instructions": []}]
    export_recipes_to_markdown(recipes, output_dir=tmp_path, include_raw_text=False)

    content = (tmp_path / "besciamella.md").read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert 'title: "Besciamella"' in content


def test_export_duplicate_titles_deduplicated(tmp_path: Path):
    """Two recipes with the same title get distinct filenames (_1 suffix)."""
    recipes = [
        {"title": "Same", "ingredients": [], "instructions": []},
        {"title": "Same", "ingredients": [], "instructions": []},
    ]
    export_recipes_to_markdown(recipes, output_dir=tmp_path, include_raw_text=False)

    files = sorted(f.name for f in tmp_path.glob("same*.md"))
    assert files == ["same.md", "same_1.md"]


def test_export_empty_list_is_noop(tmp_path: Path):
    """Passing an empty list writes no files and does not raise."""
    export_recipes_to_markdown([], output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_export_wrong_type_raises():
    """Passing a non-list raises ValueError immediately."""
    with pytest.raises(ValueError):
        export_recipes_to_markdown("not a list")  # type: ignore[arg-type]


@pytest.mark.skipif(
    not PREPARAZIONI_PATH.exists(),
    reason="Real source file not present",
)
def test_round_trip_parse_then_export(tmp_path: Path):
    """Full pipeline: parse real file → export → verify 18 files written."""
    recipes = parse_preparazioni_md(PREPARAZIONI_PATH)
    export_recipes_to_markdown(recipes, output_dir=tmp_path, include_raw_text=False)

    files = list(tmp_path.glob("*.md"))
    assert len(files) == 18
    assert (tmp_path / "besciamella.md").exists()
