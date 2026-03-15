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
        └─ 6-step line state machine (see invariants below)
              └─ List[Dict]
                    └─ export_recipes_to_markdown()
                          ├─ sanitize_filename()  — per recipe
                          └─ one .md file per recipe
```

**Public API:**
- `parse_preparazioni_md(filepath: str | Path) -> List[Dict[str, Any]]`
- `export_recipes_to_markdown(recipes, output_dir, include_raw_text=True, include_pages=True) -> None`
- `sanitize_filename(title: str, max_length: int = 80) -> str`

## Key invariants — extractor state machine

The line tokeniser in `parse_preparazioni_md` processes lines in this priority
order. **The order is load-bearing — do not reorder steps 4 and 5.**

| Step | Condition | Action |
|---|---|---|
| 1 | `--- pagina N ---` | Update `current_page`; no content |
| 2 | Blank line | Flush pending instruction buffer |
| 3 | `INGREDIENTI` keyword | Set `in_ingredients = True` |
| 4 | `in_ingredients=True` (**before title check**) | ALL-CAPS → ingredient; mixed-case → start instructions |
| 5 | `_is_recipe_title()` passes | Close current recipe, open new one |
| 6 | Anything else | Append to instruction buffer |

**Why step 4 must precede step 5:** bare ALL-CAPS words such as `SALE`, `PEPE`,
`OLIO` appear as the final ingredient in many recipes (no quantity, no comma).
`_is_recipe_title()` would match them (ALL-CAPS, no units) and falsely start a
new recipe. Checking `in_ingredients` first prevents this.

**`_is_recipe_title` heuristic — a line is a recipe title iff:**
1. Non-empty after strip
2. Not equal to `INGREDIENTI` or the document's `titolo:` field
3. Not a `--- pagina N ---` marker
4. Fully uppercase (`stripped == stripped.upper()`)
5. Does not contain a digit or measurement token (`G`, `DL`, `KG`, `ML`, `CL`, `CUCCHIAI`, `CUCCHIAINO`, `RAMETTO`, `GRANO`, `GRANI`, `PRESA`)
6. Does not end with `,`

## Recipe dict schema

Both pipelines produce dicts with these keys (all optional except `title`):

| Key | Type | Source |
|---|---|---|
| `title` | `str` | Recipe name, Title-cased |
| `ingredients` | `List[str]` | One entry per ingredient line |
| `instructions` | `List[str]` | One entry per paragraph / sentence |
| `pages` | `List[int]` | Page numbers the recipe spans |
| `raw_text` | `str` | Original source text |
| `confidence` | `float` | 0.0–1.0; only set by `chefai.parser` |
| `prep_time` | `str` | e.g. `"10 min"` — optional |
| `cook_time` | `str` | e.g. `"1 ora"` — optional |
| `difficulty` | `str` | e.g. `"Facile"` — optional |
| `servings` | `str \| None` | e.g. `"4"` — optional |

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

## Logging

All modules use `logging.getLogger(__name__)`. `extractor.py` calls
`logging.basicConfig(level=INFO)` at module level (convenience for scripts);
callers can override with `force=True`. `parser.py` does the same.

## Running tests

```bash
# from project root with venv active
venv/Scripts/python.exe -m pytest tests/ -v
```

21 tests covering `sanitize_filename`, `parse_preparazioni_md` (unit +
integration against real file), `export_recipes_to_markdown` (round-trip,
dedup, type guard, empty input), and `__version__`.

The real-file integration tests (`test_parse_real_file_*`,
`test_round_trip_parse_then_export`) are skipped automatically when
`data/processed/preparazioni-di-base.md` is absent.

## User guide

`extractor_guide.ipynb` — runnable notebook covering all three public
functions in `chefai.extractor` with synthetic and real-file examples.
Run from the project root so `REAL_FILE = Path("data/processed/…")` resolves.
