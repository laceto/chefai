# chefai

Extract, parse, and export Italian cookbook recipes from PDF and preprocessed markdown sources.

Two independent pipelines:

```
PDF ──► chefai.parser ──► ricette.json
                                │
preprocessed .md ──► chefai.extractor ──► recipes_md/*.md
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

The JSON list contains one dict per recipe:

```json
{
  "title": "Besciamella",
  "prep_time": "10 min",
  "cook_time": "15 min",
  "difficulty": "Facile",
  "servings": null,
  "ingredients": ["40 g burro", "50 g farina", "5 dl latte"],
  "instructions": ["Fate fondere il burro...", "Unite il latte..."],
  "pages": [16],
  "raw_text": "...",
  "confidence": 1.0
}
```

`confidence` is a float in `[0, 1]` that falls below `1.0` when structural
signals are weak (missing ingredients, fewer than two instruction steps).

### Pipeline 2 — Preprocessed markdown to per-recipe files (`chefai.extractor`)

Parse a preprocessed cookbook section and write one `.md` file per recipe:

```python
from chefai.extractor import parse_preparazioni_md, export_recipes_to_markdown

recipes = parse_preparazioni_md("data/processed/preparazioni-di-base.md")
export_recipes_to_markdown(recipes, output_dir="recipes_md")
```

Each output file contains YAML front-matter plus `## Ingredienti` and
`## Procedimento` sections:

```markdown
---
title: "Besciamella"
pages: [16]
---

# Besciamella

## Ingredienti
- 40 G DI BURRO CHIARIFICATO
- 50 G DI FARINA
- 5 DL DI LATTE
- SALE

## Procedimento
1. Fate fondere il burro e unite la farina...
```

#### `parse_preparazioni_md(filepath)`

| Arg | Type | Description |
|---|---|---|
| `filepath` | `str \| Path` | Path to a preprocessed section markdown file |

Returns `List[Dict]` — each dict has keys `title`, `ingredients`, `instructions`,
`pages`, `raw_text`.

Raises `FileNotFoundError` if the file does not exist; `ValueError` if no
` ```text ``` ` block is found inside it.

#### `export_recipes_to_markdown(recipes, output_dir, include_raw_text, include_pages)`

| Arg | Type | Default | Description |
|---|---|---|---|
| `recipes` | `List[Dict]` | — | Output of `parse_preparazioni_md` (or any compatible list) |
| `output_dir` | `str \| Path` | `"recipes_md"` | Directory to write files into (created if absent) |
| `include_raw_text` | `bool` | `True` | Append `## Testo originale` fenced block |
| `include_pages` | `bool` | `True` | Write `pages:` field in YAML front-matter |

Raises `ValueError` if `recipes` is not a list; `OSError` if `output_dir`
collides with an existing regular file. Duplicate slugs get `_1`, `_2`, ...
suffixes.

#### `sanitize_filename(title, max_length=80)`

Converts a recipe title to a filesystem-safe slug (accent-stripped, lowercase,
hyphen-separated). Returns the sentinel `"ricetta-senza-titolo"` for empty or
symbol-only inputs.

## Development

```bash
pip install -e ".[dev]"
pytest
```

See [`extractor_guide.ipynb`](extractor_guide.ipynb) for runnable examples of
the markdown pipeline.

## Project structure

```
chefai/
├── chefai/
│   ├── __init__.py          # __version__ = "0.1.0"
│   ├── parser.py            # PDF pipeline (requires PyMuPDF)
│   └── extractor.py         # Markdown pipeline
├── data/
│   └── processed/           # Preprocessed cookbook section .md files (gitignored)
├── tests/
│   └── test_chefai.py       # pytest suite (21 tests)
├── extractor_guide.ipynb    # User guide for chefai.extractor
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

MIT
