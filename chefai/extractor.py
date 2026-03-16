import json
import re
from pathlib import Path
from typing import List, Dict, Any
import unicodedata
import logging

# Set up logging to actually see the output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parser for "preparazioni di base" markdown files
# ---------------------------------------------------------------------------

# Measurement-unit tokens that appear only in ingredient lines, not titles.
_UNIT_TOKENS = re.compile(
    r"\b(\d+|G|DL|ML|CL|KG|CUCCHIAI[O]?|CUCCHIAINO|RAMETT[OI]|GRANO|GRANI|PRESA)\b"
)
_PAGE_MARKER = re.compile(r"^---\s*pagina\s*(\d+)\s*---\s*$", re.IGNORECASE)
_RAW_TEXT_BLOCK = re.compile(r"```text\n(.*?)```", re.DOTALL)

# Matches "INGREDIENTI" and "INGREDIENTI PER 4 PERSONE" variants.
_INGREDIENTI_HEADER = re.compile(r"^INGREDIENTI\b")
# Extracts the servings count from "INGREDIENTI PER 4 PERSONE".
_SERVINGS_PATTERN = re.compile(r"\bPER\s+(\d+)\s+PERSON[EAI]\b", re.IGNORECASE)

# Matches lines that carry recipe metadata (prep/cook time, difficulty).
# These appear between the recipe title and the INGREDIENTI line.
_METADATA_LINE = re.compile(r"^(PREPARAZIONE|COTTURA|DIFFICOLT[AÀ])\s*:", re.IGNORECASE)


def _extract_metadata(line: str, recipe: dict) -> None:
    """Parse a metadata line and write prep_time / cook_time / difficulty into *recipe*.

    Handles single-key and multi-key formats on the same line:
    - ``"PREPARAZIONE: 20 MINUTI, COTTURA: NO, DIFFICOLTÀ: *"``
    - ``"PREPARAZIONE: 45 MINUTI, COTTURA: 20 MINUTI DIFFICOLTÀ: **"``
    - ``"DIFFICOLTÀ: *"``

    Uses ``setdefault`` so the first value found wins (handles split lines where
    PREPARAZIONE appears on a previous line before COTTURA/DIFFICOLTÀ).
    """
    m = re.search(
        r"PREPARAZIONE\s*:\s*(.+?)(?=\s*,?\s*(?:COTTURA|DIFFICOLT[AÀ])\s*:|$)",
        line, re.IGNORECASE,
    )
    if m:
        recipe.setdefault("prep_time", m.group(1).strip().rstrip(",").strip())

    m = re.search(
        r"COTTURA\s*:\s*(.+?)(?=\s*,?\s*DIFFICOLT[AÀ]\s*:|$)",
        line, re.IGNORECASE,
    )
    if m:
        recipe.setdefault("cook_time", m.group(1).strip().rstrip(",").strip())

    m = re.search(r"DIFFICOLT[AÀ]\s*:\s*(.+?)$", line, re.IGNORECASE)
    if m:
        recipe.setdefault("difficulty", m.group(1).strip())


def _is_recipe_title(line: str, doc_title: str) -> bool:
    """
    Return True when *line* is an ALL-CAPS recipe title.

    Invariants:
    - Must be ALL-CAPS (after strip).
    - Must not be the document-level section heading (e.g. "PREPARAZIONI DI BASE").
    - Must not be the keyword "INGREDIENTI".
    - Must not contain digits or measurement-unit tokens (those belong to
      ingredient lines, not titles).
    - Must not end with a comma (ingredient continuation marker).
    """
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in {"INGREDIENTI", doc_title.upper()}:
        return False
    if _PAGE_MARKER.match(stripped):
        return False
    if ":" in stripped:                # KEY: VALUE lines (metadata) are never titles
        return False
    if stripped != stripped.upper():   # mixed-case → instructions
        return False
    if _UNIT_TOKENS.search(stripped):  # contains quantities / units
        return False
    if stripped.endswith(","):         # ingredient continuation
        return False
    return True


def parse_preparazioni_md(filepath: str | Path) -> List[Dict[str, Any]]:
    """
    Parse a 'preparazioni di base' markdown file into recipe dicts.

    **Expected file format**::

        ---
        titolo: SECTION TITLE
        pagine: X–Y
        ---
        ```text
        --- pagina N ---
        RECIPE TITLE
        INGREDIENTI
        AMOUNT INGREDIENT,
        AMOUNT INGREDIENT
        Mixed-case instruction prose...
        ```

    **Returns** a list of dicts compatible with :func:`export_recipes_to_markdown`::

        {
            "title":        str,          # Title-cased recipe name
            "ingredients":  List[str],    # One entry per ingredient
            "instructions": List[str],    # One entry per instruction paragraph
            "pages":        List[int],    # Page numbers where this recipe appears
            "raw_text":     str,          # Original text block for this recipe
        }

    **Failure modes**:
    - Raises :class:`FileNotFoundError` if *filepath* does not exist.
    - Raises :class:`ValueError` if no ` ```text ``` ` block is found.
    - Emits a ``WARNING`` log for any recipes with no ingredients or instructions
      (indicates a parsing anomaly worth investigating).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_content = path.read_text(encoding="utf-8")

    # --- Extract section title from frontmatter (used to skip header line) ---
    fm_title_match = re.search(r"^titolo:\s*(.+)$", raw_content, re.MULTILINE)
    doc_title = fm_title_match.group(1).strip() if fm_title_match else ""

    # --- Extract the embedded raw-text block ---
    block_match = _RAW_TEXT_BLOCK.search(raw_content)
    if not block_match:
        raise ValueError(f"No ```text block found in {path.name!r}")

    raw_block = block_match.group(1)

    # --- Tokenise lines, tracking page numbers ---
    recipes: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    current_page: int | None = None
    in_ingredients: bool = False
    pending_instruction_lines: List[str] = []  # buffer for current paragraph

    def _flush_instruction_paragraph() -> None:
        """Append buffered instruction lines as one numbered step."""
        if pending_instruction_lines and current is not None:
            current["instructions"].append(" ".join(pending_instruction_lines))
            pending_instruction_lines.clear()

    def _add_page(page: int) -> None:
        if current is not None and page not in current["pages"]:
            current["pages"].append(page)

    for line in raw_block.splitlines():
        stripped = line.strip()

        # 1. Page marker — update state, do not emit content
        pm = _PAGE_MARKER.match(stripped)
        if pm:
            current_page = int(pm.group(1))
            continue

        # 2. Blank line → paragraph break inside instructions
        if not stripped:
            _flush_instruction_paragraph()
            continue

        # 3. INGREDIENTI header — enter ingredient collection mode.
        #    Matches bare "INGREDIENTI" and "INGREDIENTI PER 4 PERSONE" variants.
        if _INGREDIENTI_HEADER.match(stripped):
            in_ingredients = True
            if current:
                current["raw_text"] += f"\n{stripped}"
                m = _SERVINGS_PATTERN.search(stripped)
                if m:
                    current.setdefault("servings", m.group(1))
            continue

        # 3.5. Metadata line (PREPARAZIONE / COTTURA / DIFFICOLTÀ).
        #      These lines appear between the recipe title and INGREDIENTI in
        #      most files.  Extract structured fields and skip the raw text from
        #      ingredients/instructions.
        if _METADATA_LINE.match(stripped):
            if current is not None:
                current["raw_text"] += f"\n{stripped}"
                _extract_metadata(stripped, current)
            continue

        # 4. Ingredient-mode content:
        #    MUST be evaluated before _is_recipe_title so that bare ALL-CAPS
        #    words (e.g. SALE, PEPE) that happen to look like valid titles are
        #    treated as the final ingredient in the list, not as new recipes.
        if in_ingredients and current is not None:
            if current_page is not None:
                _add_page(current_page)
            current["raw_text"] += f"\n{stripped}"
            if stripped == stripped.upper():
                # Still ALL-CAPS → another ingredient (strip trailing comma)
                current["ingredients"].append(stripped.rstrip(","))
            else:
                # First mixed-case line → transition to instructions
                in_ingredients = False
                pending_instruction_lines.append(stripped)
            continue

        # 5. Recipe title (only reached when NOT in ingredient mode)
        if _is_recipe_title(stripped, doc_title):
            _flush_instruction_paragraph()
            if current:
                if not current["ingredients"]:
                    logger.warning("Recipe %r has no ingredients — check source", current["title"])
                if not current["instructions"]:
                    logger.warning("Recipe %r has no instructions — check source", current["title"])
                recipes.append(current)

            current = {
                "title": stripped.title(),
                "ingredients": [],
                "instructions": [],
                "pages": [current_page] if current_page is not None else [],
                "raw_text": stripped,   # seed; extended below
            }
            in_ingredients = False
            pending_instruction_lines.clear()
            continue

        # 6. Instruction / other prose — skip orphan lines before any recipe
        if current is None:
            continue

        if current_page is not None:
            _add_page(current_page)
        current["raw_text"] += f"\n{stripped}"
        pending_instruction_lines.append(stripped)

    # Finalise last open recipe
    _flush_instruction_paragraph()
    if current:
        if not current["ingredients"]:
            logger.warning("Recipe %r has no ingredients — check source", current["title"])
        if not current["instructions"]:
            logger.warning("Recipe %r has no instructions — check source", current["title"])
        recipes.append(current)

    logger.info("Parsed %d recipes from %s", len(recipes), path.name)
    return recipes

def sanitize_filename(title: str, max_length: int = 80) -> str:
    """Converte titolo in nome file valido e leggibile"""
    title = str(title) # Ensure string
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    title = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower())
    title = re.sub(r'\s+', '-', title.strip())
    title = re.sub(r'-+', '-', title)
    
    return (title[:max_length].rstrip('-')) or "ricetta-senza-titolo"

def export_recipes_to_markdown(
    recipes: List[Dict[str, Any]],
    output_dir: str | Path = "recipes_md",
    include_raw_text: bool = True,
    include_pages: bool = True
) -> None:
    # --- FAIL FAST CHECKS ---
    if not isinstance(recipes, list):
        raise ValueError(f"Expected list of recipes, got {type(recipes)}")
    
    if not recipes:
        logger.warning("No recipes provided to export.")
        return

    out_dir = Path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise OSError(f"Cannot create directory '{out_dir}' because a file with that name exists.")

    for idx, recipe in enumerate(recipes):
        # Fallback title if missing
        title = recipe.get("title", "").strip() or f"Ricetta_{idx+1}"
        
        filename = sanitize_filename(title) + ".md"
        filepath = out_dir / filename

        # Unique filename logic
        counter = 1
        original_stem = filepath.stem
        while filepath.exists():
            filepath = out_dir / f"{original_stem}_{counter}.md"
            counter += 1

        content_lines = []

        # --- Front-matter YAML (Safe formatting) ---
        content_lines.append("---")
        # Use simple escaping for YAML values to prevent syntax errors
        safe_title = title.replace('"', '\\"')
        content_lines.append(f'title: "{safe_title}"')
        
        # Helper to add YAML fields safely
        for key in ["prep_time", "cook_time", "difficulty", "servings"]:
            val = recipe.get(key)
            if val and val != "N/A":
                content_lines.append(f'{key}: "{str(val).replace('"', "")}"')
        
        if include_pages and recipe.get("pages"):
            content_lines.append(f"pages: [{', '.join(map(str, recipe['pages']))}]")
            
        conf = recipe.get("confidence", 1.0)
        if conf < 0.9:
            content_lines.append(f"confidence: {conf:.2f}")
        content_lines.append("---")
        content_lines.append(f"\n# {title}\n")

        # --- Body Content ---
        if any(recipe.get(k) for k in ["prep_time", "cook_time", "difficulty", "servings"]):
            content_lines.append("### Informazioni")
            if recipe.get("prep_time"): content_lines.append(f"- **Prep:** {recipe['prep_time']}")
            if recipe.get("cook_time"): content_lines.append(f"- **Cottura:** {recipe['cook_time']}")
            if recipe.get("difficulty"): content_lines.append(f"- **Difficoltà:** {recipe['difficulty']}")
            content_lines.append("")

        if recipe.get("ingredients"):
            content_lines.append("## Ingredienti")
            content_lines.extend([f"- {ing}" for ing in recipe["ingredients"]])
            content_lines.append("")

        if recipe.get("instructions"):
            content_lines.append("## Procedimento")
            content_lines.extend([f"{i}. {step}" for i, step in enumerate(recipe["instructions"], 1)])
            content_lines.append("")

        if include_raw_text and recipe.get("raw_text"):
            content_lines.append("## Testo originale")
            content_lines.append("```text")
            # Sanitize internal backticks to prevent breaking the code block
            raw = recipe["raw_text"].replace("```", "'''")
            content_lines.append(raw)
            content_lines.append("```")

        # --- Writing (Atomic-like) ---
        try:
            filepath.write_text("\n".join(content_lines), encoding="utf-8")
            logger.info(f"Successo: {filepath.name}")
        except Exception as e:
            logger.error(f"Errore critico durante la scrittura di {filepath}: {e}")


def export_recipes_to_json(
    recipes: List[Dict[str, Any]],
    output_dir: str | Path = "data/recipes",
    include_raw_text: bool = True,
) -> None:
    """
    Write one JSON file per recipe into *output_dir*.

    Each file is named after the recipe slug (via :func:`sanitize_filename`)
    with a ``.json`` extension.  Duplicate slugs receive ``_1``, ``_2``, …
    suffixes — identical logic to :func:`export_recipes_to_markdown`.

    **Args:**
    - ``recipes`` — list of recipe dicts (output of :func:`parse_preparazioni_md`
      or any compatible source).
    - ``output_dir`` — destination directory; created if absent.
    - ``include_raw_text`` — when ``False``, the ``raw_text`` key is omitted
      from every written dict (keeps files small).

    **Raises:**
    - ``ValueError`` if *recipes* is not a list.
    - ``OSError`` if *output_dir* exists as a regular file.

    **Invariant:** the JSON written is a single object (not an array), encoded
    UTF-8, indented with 2 spaces.  The field order matches insertion order of
    the source dict, with ``raw_text`` always last when present.
    """
    if not isinstance(recipes, list):
        raise ValueError(f"Expected list of recipes, got {type(recipes)}")

    if not recipes:
        logger.warning("No recipes provided to export.")
        return

    out_dir = Path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise OSError(
            f"Cannot create directory '{out_dir}' because a file with that name exists."
        )

    for idx, recipe in enumerate(recipes):
        title = recipe.get("title", "").strip() or f"Ricetta_{idx + 1}"

        filepath = out_dir / (sanitize_filename(title) + ".json")
        counter = 1
        original_stem = filepath.stem
        while filepath.exists():
            filepath = out_dir / f"{original_stem}_{counter}.json"
            counter += 1

        # Optionally strip raw_text to keep files lean
        payload = {k: v for k, v in recipe.items() if include_raw_text or k != "raw_text"}

        try:
            filepath.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"Successo: {filepath.name}")
        except Exception as e:
            logger.error(f"Errore critico durante la scrittura di {filepath}: {e}")