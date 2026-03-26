# chefai

Extract, parse, and export Italian cookbook recipes from PDF and preprocessed markdown sources.

Three independent pipelines:

```
PDF ──► chefai.parser ──► ricette.json

preprocessed .md ──► chefai.extractor ──► per-recipe .md files
                              │
                              ▼
                    scripts/build_corpus.py ──► data/recipes.json

RSS feed TSV files ──► chefai.feed_embedder ──► FAISS vectorstore
  (via embed_recipes.py)                         + feeds_registry.tsv
```

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -e ".[dev]"
```

`chefai.parser` additionally requires **PyMuPDF**:

```bash
pip install pymupdf tqdm
```

## Usage

### Pipeline 1 — PDF to JSON (`chefai.parser`)

Extract recipes from a cookbook PDF into a structured JSON file:

```python
from chefai.parser import extract_recipes_from_pdf

extract_recipes_from_pdf("libro.pdf", output_json="ricette.json")
# writes ricette.json with a list of recipe dicts
```

`confidence` is a float in `[0, 1]` that falls below `1.0` when structural
signals are weak (missing ingredients, fewer than two instruction steps).

### Pipeline 2 — Preprocessed markdown (`chefai.extractor`)

#### Parse a single section file

```python
from chefai.extractor import parse_preparazioni_md, export_recipes_to_markdown

recipes = parse_preparazioni_md("data/processed/antipasti-di-mare.md")
export_recipes_to_markdown(recipes, output_dir="recipes_md")
```

Works with all 11 section files in `data/processed/`. Each output file
contains YAML front-matter plus `## Ingredienti` and `## Procedimento` sections.

#### Build the full corpus (all section files → one JSON)

```bash
python scripts/build_corpus.py
# writes data/recipes.json  (atomic write; overwrites on rerun)

python scripts/build_corpus.py --no-raw-text   # smaller file
python scripts/build_corpus.py --output data/my_corpus.json
```

Or from Python:

```python
from scripts.build_corpus import build_corpus
from pathlib import Path

recipes = build_corpus(
    processed_dir=Path("data/processed"),
    output_path=Path("data/recipes.json"),
    include_raw_text=False,
)
print(len(recipes), "recipes written")
```

Every recipe in the corpus carries a `source_file` field (e.g.
`"antipasti-di-mare"`) so its origin is traceable.

#### Export a list of recipes to JSON (per-file or single-file)

```python
from chefai.extractor import export_recipes_to_json, export_corpus_to_json

# One .json file per recipe (legacy; useful for per-recipe inspection)
export_recipes_to_json(recipes, output_dir="data/recipes")

# One combined array file (preferred for downstream consumers)
export_corpus_to_json(recipes, output_path="data/recipes.json")
```

### Pipeline 3 — FAISS vectorstore (`chefai.feed_embedder`)

Builds and incrementally updates a FAISS vectorstore from RSS feed articles
using the OpenAI Batch API (50% cheaper than synchronous calls via kitai).

```bash
# cold start — embeds all feed articles and creates the store from scratch
python embed_recipes.py

# incremental — embeds only new guids and appends to the existing store
python embed_recipes.py

# set LOG_LEVEL=DEBUG for detailed batch polling output
LOG_LEVEL=DEBUG python embed_recipes.py
```

Requires `OPENAI_API_KEY` in the environment (loaded from `.env`).

All reusable logic lives in `chefai.feed_embedder`; import from there for
programmatic use:

```python
from chefai.feed_embedder import (
    load_registry, save_registry,
    load_all_feed_files, find_new_articles, assign_ids,
    build_documents,
    run_embedding_batch, align_pairs_to_docs,
    init_vectorstore, update_vectorstore,
    DEFAULT_EMBED_MODEL, DEFAULT_EMBED_DIMS,
)
```

### Recipe dict schema

All pipelines produce dicts with these keys (all optional except `title`):

| Key | Type | Notes |
|---|---|---|
| `title` | `str` | Title-cased, e.g. `"Brodo Di Carne"` |
| `ingredients` | `List[str]` | One entry per ingredient; multi-ingredient lines are split |
| `instructions` | `List[str]` | One entry per blank-line-separated paragraph |
| `pages` | `List[int]` | Source page numbers |
| `raw_text` | `str` | Original lines from the source text block |
| `prep_time` | `str` | e.g. `"20 MINUTI"` — from `PREPARAZIONE:` metadata line |
| `cook_time` | `str` | e.g. `"NO"` or `"30 MINUTI"` — from `COTTURA:` metadata line |
| `difficulty` | `str` | e.g. `"*"` or `"**"` — from `DIFFICOLTÀ:` metadata line |
| `servings` | `str` | e.g. `"4"` — from `INGREDIENTI PER 4 PERSONE` header |
| `confidence` | `float` | `0.0–1.0`; only set by `chefai.parser` |
| `source_file` | `str` | Source file stem — set by `scripts/build_corpus.py` only |

## API reference — `chefai.extractor`

#### `parse_preparazioni_md(filepath)`

| Arg | Type | Description |
|---|---|---|
| `filepath` | `str \| Path` | Path to a preprocessed section markdown file |

Returns `List[Dict]`. Handles both `INGREDIENTI` (bare) and
`INGREDIENTI PER N PERSONE` variants. Parses `PREPARAZIONE:`,
`COTTURA:`, and `DIFFICOLTÀ:` metadata lines into structured fields.

Raises `FileNotFoundError` if the file does not exist; `ValueError` if no
` ```text ``` ` block is found inside it.

#### `export_recipes_to_markdown(recipes, output_dir, include_raw_text, include_pages)`

| Arg | Type | Default | Description |
|---|---|---|---|
| `recipes` | `List[Dict]` | — | Output of `parse_preparazioni_md` or compatible list |
| `output_dir` | `str \| Path` | `"recipes_md"` | Directory to write files into (created if absent) |
| `include_raw_text` | `bool` | `True` | Append `## Testo originale` fenced block |
| `include_pages` | `bool` | `True` | Write `pages:` in YAML front-matter |

Raises `ValueError` if `recipes` is not a list; `OSError` if `output_dir`
collides with an existing regular file. Duplicate slugs get `_1`, `_2`, …
suffixes.

#### `export_recipes_to_json(recipes, output_dir, include_raw_text)`

Writes one `.json` file per recipe into `output_dir`. Same deduplication
and error behaviour as `export_recipes_to_markdown`.

#### `export_corpus_to_json(recipes, output_path, include_raw_text)`

Writes the entire list as a single JSON array to `output_path`. Uses an
atomic write (`.tmp` + rename) so a failed run never leaves a corrupt file.
Raises `OSError` if `output_path` is an existing directory.

#### `sanitize_filename(title, max_length=80)`

Converts a recipe title to a filesystem-safe slug (accent-stripped,
lowercase, hyphen-separated). Returns `"ricetta-senza-titolo"` for empty
or symbol-only inputs.

## Development

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
pytest
```

74 tests across two suites:
- `tests/test_chefai.py` — 54 tests covering `chefai.parser` / `chefai.extractor`
- `tests/test_feed_embedder.py` — 20 tests covering `chefai.feed_embedder` (kitai.batch mocked)

See [`extractor_guide.ipynb`](extractor_guide.ipynb) for runnable examples of `chefai.extractor`.

## Project structure

```
chefai/
├── chefai/
│   ├── __init__.py          # __version__ = "0.1.0"
│   ├── parser.py            # PDF pipeline (requires PyMuPDF)
│   ├── extractor.py         # Markdown pipeline
│   └── feed_embedder.py     # FAISS vectorstore helpers (RSS feed embeddings)
├── scripts/
│   └── build_corpus.py      # Batch-parse all section files → data/recipes.json
├── data/
│   └── processed/           # Preprocessed cookbook section .md files (gitignored)
├── tests/
│   ├── test_chefai.py       # pytest suite (54 tests — parser + extractor)
│   └── test_feed_embedder.py# pytest suite (20 tests — feed_embedder)
├── embed_recipes.py         # Entry-point: build/update FAISS vectorstore from feeds
├── constants.py             # Shared file paths (VECTORSTORE_DIR, RAW_FEED_DIR, …)
├── extractor_guide.ipynb    # User guide for chefai.extractor
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

MIT
