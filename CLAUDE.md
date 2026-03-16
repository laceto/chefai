# CLAUDE.md — chefai architecture notes

Reference for AI assistants working on this codebase.

## Package layout

```
chefai/
├── __init__.py      __version__ = "0.1.0"
├── parser.py        PDF pipeline — requires PyMuPDF (fitz) and tqdm
└── extractor.py     Markdown pipeline — stdlib only
```

## Two independent pipelines

### PDF pipeline (`chefai.parser`)

```
fitz.open(pdf_path)
  └─ per-page text → Dict[int, str]
        └─ extract_recipes_from_pages()
              ├─ is_likely_title()      — detects recipe boundaries
              ├─ is_continuation()      — determines page-break continuation
              ├─ parse_ingredients()    — regex block extractor
              ├─ split_instructions()   — sentence splitter
              └─ extract_key_value_block() — pulls prep/cook/difficulty
        └─ List[Dict]  (Recipe dataclasses → asdict)
              └─ written to output_json via json.dump
```

**Public API:**
- `extract_recipes_from_pdf(pdf_path, output_json="ricette.json") -> None`
- `extract_recipes_from_pages(page_texts: Dict[int, str]) -> List[Dict]`

### Markdown pipeline (`chefai.extractor`)

```
.md file (preprocessed section)
  └─ parse_preparazioni_md()
        ├─ reads frontmatter titolo: field
        ├─ extracts ```text ... ``` block
        └─ 7-step line state machine (see invariants below)
              └─ List[Dict]
                    └─ export_recipes_to_markdown()
                          ├─ sanitize_filename()  — per recipe
                          └─ one .md file per recipe
```

**Public API:**
- `parse_preparazioni_md(filepath: str | Path) -> List[Dict[str, Any]]`
- `export_recipes_to_markdown(recipes, output_dir, include_raw_text=True, include_pages=True) -> None`
- `export_recipes_to_json(recipes, output_dir, include_raw_text=True) -> None`
- `export_corpus_to_json(recipes, output_path, include_raw_text=True) -> None`
- `sanitize_filename(title: str, max_length: int = 80) -> str`

## Key invariants — extractor state machine

The line tokeniser in `parse_preparazioni_md` processes lines in this priority
order. **The order is load-bearing — do not reorder steps 3.5, 4, and 5.**

| Step | Condition | Action |
|---|---|---|
| 1 | `--- pagina N ---` | Update `current_page`; no content |
| 2 | Blank line | Flush pending instruction buffer |
| 3 | `^INGREDIENTI\b` (bare or `PER N PERSONE`) | Set `in_ingredients = True`; extract `servings` |
| 3.5 | `^(PREPARAZIONE\|COTTURA\|DIFFICOLT[AÀ])\s*:` | Extract `prep_time` / `cook_time` / `difficulty`; skip line |
| 4 | `in_ingredients=True` (**before title check**) | ALL-CAPS → ingredient; mixed-case → start instructions |
| 5 | `_is_recipe_title()` passes | Close current recipe, open new one |
| 6 | Anything else | Append to instruction buffer |

**Why step 4 must precede step 5:** bare ALL-CAPS words such as `SALE`, `PEPE`,
`OLIO` appear as the final ingredient in many recipes (no quantity, no comma).
`_is_recipe_title()` would match them (ALL-CAPS, no units) and falsely start a
new recipe. Checking `in_ingredients` first prevents this.

**Why step 3.5 must precede step 4:** metadata lines like `DIFFICOLTÀ: *` are
ALL-CAPS with no digits or units, so `_is_recipe_title()` would match them
and open a spurious recipe. The `":"` guard in `_is_recipe_title` is the primary
defence, but step 3.5 also ensures these lines are never seen by step 4.

**`_is_recipe_title` heuristic — a line is a recipe title iff:**
1. Non-empty after strip
2. Not equal to `INGREDIENTI` or the document's `titolo:` field
3. Not a `--- pagina N ---` marker
4. Does not contain `:` (rules out all `KEY: VALUE` metadata lines)
5. Fully uppercase (`stripped == stripped.upper()`)
6. Does not contain a digit or measurement token (`G`, `DL`, `KG`, `ML`, `CL`, `CUCCHIAI`, `CUCCHIAINO`, `RAMETTO`, `GRANO`, `GRANI`, `PRESA`)
7. Does not end with `,`

**Step 4 ingredient splitting:** ALL-CAPS ingredient lines are passed through
`_split_ingredient_line(stripped)` before being added to the recipe. This
function uses `_INGREDIENT_SPLIT_BOUNDARY` (a comma followed by a digit or
known unit token) to detect multi-ingredient lines (e.g.
`"20 ACCIUGHE, 1 TUORLO D'UOVO,"` → two entries). Single-ingredient lines
are returned unchanged. **The unit token lists in `_UNIT_TOKENS` and
`_INGREDIENT_SPLIT_BOUNDARY` must be kept in sync when adding new tokens.**

## Recipe dict schema

Both pipelines produce dicts with these keys (all optional except `title`):

| Key | Type | Source |
|---|---|---|
| `title` | `str` | Recipe name, Title-cased |
| `ingredients` | `List[str]` | One entry per ingredient (lines split by `_split_ingredient_line`) |
| `instructions` | `List[str]` | One entry per paragraph / sentence |
| `pages` | `List[int]` | Page numbers the recipe spans |
| `raw_text` | `str` | Original source text |
| `confidence` | `float` | 0.0–1.0; only set by `chefai.parser` |
| `prep_time` | `str` | e.g. `"30 MINUTI"` — optional; from metadata line |
| `cook_time` | `str` | e.g. `"NO"` or `"20 MINUTI"` — optional; from metadata line |
| `difficulty` | `str` | e.g. `"*"` or `"**"` — optional; from metadata line |
| `servings` | `str` | e.g. `"4"` — optional; from `INGREDIENTI PER N PERSONE` |
| `source_file` | `str` | Stem of source `.md` file — set by `scripts/build_corpus.py` only |

## Preprocessed markdown format

Files in `data/processed/` follow this structure:

```markdown
---
titolo: SECTION TITLE
pagine: X–Y
numero_pagine: N
---

# SECTION TITLE

**Pagine:** X – Y

```text
--- pagina X ---
RECIPE TITLE
INGREDIENTI
AMOUNT INGREDIENT,
AMOUNT INGREDIENT
Mixed-case instruction prose...
```
```

`parse_preparazioni_md` reads the `titolo:` frontmatter field to exclude the
section heading from recipe detection, then parses the ` ```text ``` ` block.

## Error behaviour

| Function | Condition | Result |
|---|---|---|
| `parse_preparazioni_md` | File missing | `FileNotFoundError` |
| `parse_preparazioni_md` | No ` ```text ``` ` block | `ValueError` |
| `parse_preparazioni_md` | Recipe with 0 ingredients or 0 steps | `WARNING` log, recipe still returned |
| `export_recipes_to_markdown` | `recipes` not a list | `ValueError` |
| `export_recipes_to_markdown` | `output_dir` is a regular file | `OSError` |
| `export_recipes_to_markdown` | Empty list | `WARNING` log, no-op |
| `export_recipes_to_markdown` | Duplicate slugs | appends `_1`, `_2`, … |
| `export_recipes_to_json` | `recipes` not a list | `ValueError` |
| `export_recipes_to_json` | `output_dir` is a regular file | `OSError` |
| `export_recipes_to_json` | Empty list | `WARNING` log, no-op |
| `export_recipes_to_json` | Duplicate slugs | appends `_1`, `_2`, … |
| `export_corpus_to_json` | `recipes` not a list | `ValueError` |
| `export_corpus_to_json` | `output_path` is a directory | `OSError` |
| `export_corpus_to_json` | Empty list | `WARNING` log, no-op |

## Corpus build script (`scripts/build_corpus.py`)

Discovers all `*.md` files in `data/processed/`, parses each with
`parse_preparazioni_md`, stamps a `source_file` field (path stem, e.g.
`"antipasti-di-mare"`) on every recipe dict, and writes the combined list to
`data/recipes.json` via `export_corpus_to_json` (atomic write).

```bash
# from project root with venv active
python scripts/build_corpus.py
python scripts/build_corpus.py --no-raw-text          # omit raw_text to reduce size
python scripts/build_corpus.py --output data/out.json # custom destination
```

A single malformed source file is logged as an error and skipped; the rest of
the corpus is still written.

## Logging

All modules use `logging.getLogger(__name__)`. `extractor.py` calls
`logging.basicConfig(level=INFO)` at module level (convenience for scripts);
callers can override with `force=True`. `parser.py` does the same.

## Running tests

```bash
# from project root with venv active
venv/Scripts/python.exe -m pytest tests/ -v
```

55 tests covering `sanitize_filename`, `_split_ingredient_line`,
`parse_preparazioni_md` (unit + integration against real files),
`export_recipes_to_markdown` (round-trip, dedup, type guard, empty input),
`export_recipes_to_json`, `export_corpus_to_json`, `build_corpus`, and
`__version__`.

Real-file integration tests are skipped automatically when the corresponding
source file or directory is absent:
- `test_parse_real_file_*`, `test_round_trip_parse_then_export` → `preparazioni-di-base.md`
- `test_parse_antipasti_di_mare_*` → `antipasti-di-mare.md`
- `test_build_corpus_produces_valid_json` → `data/processed/`

## User guide

`extractor_guide.ipynb` — runnable notebook covering all three public
functions in `chefai.extractor` with synthetic and real-file examples.
Run from the project root so `REAL_FILE = Path("data/processed/…")` resolves.
