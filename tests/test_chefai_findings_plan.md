# Code Review: test_chefai.py

**Review Date:** 2026-03-15
**Reviewer:** Claude Code
**File:** `tests/test_chefai.py`

## Executive Summary

The test file is a project scaffold placeholder — it contains exactly one assertion
(`__version__` is a non-empty string) and provides **zero coverage** of the two
modules that were just actively developed: `extractor.parse_preparazioni_md` and
`extractor.export_recipes_to_markdown`. This is a critical coverage gap: all the
parsing logic (title detection, ingredient/instruction splitting, multi-page tracking,
`SALE`-as-ingredient fix) is completely untested and will silently regress on future
changes. The file is not malicious or structurally wrong; it simply doesn't test
anything yet.

## Findings

### 🔴 Critical Issues (Count: 1)

#### Issue 1: Zero coverage of `extractor` — the only non-trivial module
**Severity:** Critical
**Category:** Testing / Correctness
**Lines:** 1–9 (the entire file)

**Description:**
`parse_preparazioni_md` embeds a 6-step state machine (page markers, ingredient mode,
instruction paragraphs, title heuristics). `export_recipes_to_markdown` writes files
with YAML front-matter, deduplicates filenames, and sanitises names. Neither function
has a single test. A one-line rename in `_is_recipe_title` or a changed regex would
silently break all recipe parsing with no test failure.

**Current Code:**
```python
import chefai

def test_version():
    assert isinstance(chefai.__version__, str)
    assert len(chefai.__version__) > 0
```

**Impact:**
- Any regression in `parse_preparazioni_md` (e.g., re-introducing the `SALE`-as-title
  bug) goes undetected.
- `export_recipes_to_markdown` file-write failures, YAML corruption, and duplicate
  filenames are untested.
- CI gives a false green on every push.

**Recommendation:**
Add pytest tests that exercise: happy path, failure modes, and the edge cases that
have already caused bugs (bare ALL-CAPS ingredients, multi-page recipes, missing text
block).

---

### 🟠 High Priority Issues (Count: 2)

#### Issue 2: No failure-mode tests for `parse_preparazioni_md`
**Severity:** High
**Category:** Correctness / Observability
**Lines:** N/A (missing tests)

**Description:**
`parse_preparazioni_md` raises `FileNotFoundError` and `ValueError` on bad input, and
emits `WARNING` logs for structurally malformed recipes. None of these paths are
verified — a future refactor could silently swallow errors.

**Impact:**
- Callers receive no error on missing files if the guard is accidentally removed.
- The WARNING log path (partial recipe with no ingredients) is the only observability
  signal for parsing anomalies; it must be tested to ensure it fires.

**Proposed Solution:**
```python
import pytest, logging
from chefai.extractor import parse_preparazioni_md

def test_parse_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_preparazioni_md("does_not_exist.md")

def test_parse_no_text_block(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("---\ntitolo: TEST\n---\n# No code block\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No ```text block"):
        parse_preparazioni_md(f)
```

---

#### Issue 3: `sanitize_filename` has no tests for its boundary cases
**Severity:** High
**Category:** Correctness
**Lines:** N/A (missing tests)

**Description:**
`sanitize_filename` strips accents, collapses whitespace, and enforces a max length,
but three edge cases are untested: empty string input → fallback sentinel, max-length
truncation, and strings that become empty after stripping special characters.

**Impact:**
- Empty-after-strip path returns the sentinel `"ricetta-senza-titolo"` — if the
  sentinel changes, callers break silently.
- Truncation at `max_length=80` could produce a trailing `-` which is stripped; this
  is invisible without a test.

**Proposed Solution:**
```python
from chefai.extractor import sanitize_filename

def test_sanitize_empty():
    assert sanitize_filename("") == "ricetta-senza-titolo"
    assert sanitize_filename("!!!") == "ricetta-senza-titolo"

def test_sanitize_truncation():
    long = "a" * 100
    assert len(sanitize_filename(long)) <= 80

def test_sanitize_accents():
    assert sanitize_filename("Crêpe") == "crepe"
    assert sanitize_filename("Ragù") == "ragu"
```

---

### 🟡 Medium Priority Issues (Count: 2)

#### Issue 4: No round-trip integration test (parse → export → file content)
**Severity:** Medium
**Category:** Integration / Maintainability
**Lines:** N/A (missing tests)

**Description:**
The pipeline is: `parse_preparazioni_md` → list of dicts → `export_recipes_to_markdown`
→ `.md` files. No test verifies the full chain against the real source file, so a
schema mismatch between the two functions (e.g., a key rename) would only surface at
runtime.

**Proposed Solution:**
```python
def test_round_trip(tmp_path):
    src = Path("data/processed/preparazioni-di-base.md")
    recipes = parse_preparazioni_md(src)
    export_recipes_to_markdown(recipes, output_dir=tmp_path)
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 18
    assert (tmp_path / "besciamella.md").exists()
```

---

#### Issue 5: No test for duplicate-filename deduplication in `export_recipes_to_markdown`
**Severity:** Medium
**Category:** Correctness
**Lines:** N/A (missing tests)

**Description:**
The exporter has a `while filepath.exists()` loop that appends `_1`, `_2`… suffixes
for collisions. This logic is completely untested — a refactor could change the
suffix format and no test would catch it.

**Proposed Solution:**
```python
def test_export_duplicate_titles(tmp_path):
    recipes = [{"title": "Same"}, {"title": "Same"}]
    export_recipes_to_markdown(recipes, output_dir=tmp_path, include_raw_text=False)
    files = sorted(tmp_path.glob("same*.md"))
    assert len(files) == 2
    assert files[0].name == "same.md"
    assert files[1].name == "same_1.md"
```

---

### 🟢 Low Priority Issues (Count: 1)

#### Issue 6: Test function has no docstring
**Severity:** Low
**Category:** Documentation
**Lines:** 6

**Description:**
`test_version` has no docstring. In this project docstrings on tests matter because
the `pyproject.toml` sets `mypy strict = true`, and they also appear in pytest's `-v`
output as the human-readable description of what is being verified.

**Proposed Solution:**
```python
def test_version():
    """Package exposes a non-empty __version__ string."""
    assert isinstance(chefai.__version__, str)
    assert len(chefai.__version__) > 0
```

---

## Positive Observations

- `pyproject.toml` correctly configures `testpaths`, `addopts = "-v --tb=short"`, and
  the `dev` extras group — the test infrastructure is properly scaffolded.
- The `__init__.py` is minimal and correct; `__version__` is defined in one place.
- pytest + pytest-cov are already in the dev dependencies.

## Action Plan

### Phase 1: Critical (Immediate)
- [ ] Add `test_parse_preparazioni_md_happy_path` — verify recipe count, titles, ingredient lists, page tracking
- [ ] Add `test_parse_bare_allcaps_ingredient` — regression test for the `SALE`-as-title bug
- [ ] Add `test_parse_missing_file` and `test_parse_no_text_block`

### Phase 2: High Priority (This sprint)
- [ ] Add `test_sanitize_filename_*` edge cases (empty, truncation, accents)
- [ ] Add `test_export_duplicate_titles`

### Phase 3: Medium Priority (Next sprint)
- [ ] Add full round-trip integration test against the real source file
- [ ] Add `test_export_creates_valid_yaml_frontmatter`

### Phase 4: Low Priority (Backlog)
- [ ] Add docstrings to all test functions

## Technical Debt Estimate

- **Total Issues:** 6 (1 critical, 2 high, 2 medium, 1 low)
- **Estimated Fix Time:** 2–3 hours
- **Risk Level:** High (zero coverage of the only non-trivial module)
- **Recommended Refactor:** No — add tests directly; code under test is well-structured

## References

- pytest docs — parametrize, tmp_path fixture: https://docs.pytest.org/en/stable/
- pytest `caplog` for asserting log output: https://docs.pytest.org/en/stable/how-to/logging.html
