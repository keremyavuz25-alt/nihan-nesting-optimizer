"""DXF parser for Gemini CAD Systems pattern files.

Handles the specific entity/layer conventions used by Gemini:
  - Layer 1 POLYLINE: cutlines  |  Layer 1 TEXT: metadata
  - Layer 2 POINT+TEXT: notch marks
  - Layer 3 POINT: seamline points
  - Layer 4 POINT (z=10): grainline direction markers
  - Layer 7 LINE: reference guide lines
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _tokenize(filepath: str) -> list[tuple[int, str]]:
    """Read a DXF file and return a list of (group_code, value) pairs.

    Tries windows-1254 (Turkish) first, then utf-8, then latin-1.
    """
    raw: str | None = None
    for enc in ("windows-1254", "utf-8", "latin-1"):
        try:
            raw = Path(filepath).read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if raw is None:
        raise ValueError(f"Cannot decode file: {filepath}")

    lines = raw.splitlines()
    tokens: list[tuple[int, str]] = []
    i = 0
    while i + 1 < len(lines):
        code_str = lines[i].strip()
        val_str = lines[i + 1].strip()
        try:
            code = int(code_str)
        except ValueError:
            i += 1
            continue
        tokens.append((code, val_str))
        i += 2
    return tokens


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

def _split_sections(tokens: list[tuple[int, str]]) -> dict[str, list[tuple[int, str]]]:
    """Split token stream into named sections (HEADER, BLOCKS, ENTITIES, …)."""
    sections: dict[str, list[tuple[int, str]]] = {}
    idx = 0
    n = len(tokens)
    while idx < n:
        code, val = tokens[idx]
        if code == 0 and val == "SECTION":
            idx += 1
            if idx < n and tokens[idx][0] == 2:
                sec_name = tokens[idx][1]
                idx += 1
                body: list[tuple[int, str]] = []
                while idx < n:
                    if tokens[idx] == (0, "ENDSEC"):
                        idx += 1
                        break
                    body.append(tokens[idx])
                    idx += 1
                sections[sec_name] = body
                continue
        idx += 1
    return sections


# ---------------------------------------------------------------------------
# Entity parsing helpers
# ---------------------------------------------------------------------------

def _collect_entity(tokens: list[tuple[int, str]], start: int) -> tuple[dict[int, str | list], int]:
    """Collect group codes for one entity starting at *start* (pointing at the token
    right after the (0, TYPE) pair).  Returns (code_map, next_index).

    For codes that can repeat (10, 20, etc.) inside VERTEX-style sub-entities
    this is NOT used — POLYLINE has its own flow.
    """
    codes: dict[int, str | list] = {}
    idx = start
    n = len(tokens)
    while idx < n:
        c, v = tokens[idx]
        if c == 0:
            break
        codes[c] = v
        idx += 1
    return codes, idx


def _float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Entity builders
# ---------------------------------------------------------------------------

def _parse_polyline(tokens: list[tuple[int, str]], start: int) -> tuple[dict, int]:
    """Parse a POLYLINE entity (with VERTEX + SEQEND).

    *start* points to the token after (0, "POLYLINE").
    """
    # Collect POLYLINE header codes
    codes: dict[int, str] = {}
    idx = start
    n = len(tokens)
    while idx < n:
        c, v = tokens[idx]
        if c == 0:
            break
        codes[c] = v
        idx += 1

    layer = codes.get(8, "0")
    closed = (_int(codes.get(70, "0")) & 1) == 1

    vertices: list[tuple[float, float]] = []
    while idx < n:
        c, v = tokens[idx]
        if c == 0:
            if v == "VERTEX":
                idx += 1
                vx, vy = 0.0, 0.0
                while idx < n:
                    vc, vv = tokens[idx]
                    if vc == 0:
                        break
                    if vc == 10:
                        vx = _float(vv)
                    elif vc == 20:
                        vy = _float(vv)
                    idx += 1
                vertices.append((vx, vy))
            elif v == "SEQEND":
                idx += 1
                # skip SEQEND codes
                while idx < n and tokens[idx][0] != 0:
                    idx += 1
                break
            else:
                break
        else:
            idx += 1

    entity: dict[str, Any] = {
        "type": "POLYLINE",
        "layer": layer,
        "closed": closed,
        "vertices": vertices,
    }
    return entity, idx


def _parse_point(codes: dict[int, str]) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "type": "POINT",
        "layer": codes.get(8, "0"),
        "x": _float(codes.get(10)),
        "y": _float(codes.get(20)),
    }
    z = codes.get(30)
    if z is not None:
        entity["z"] = _float(z)
    angle = codes.get(50)
    if angle is not None:
        entity["angle"] = _float(angle)
    return entity


def _parse_text(codes: dict[int, str]) -> dict[str, Any]:
    return {
        "type": "TEXT",
        "layer": codes.get(8, "0"),
        "x": _float(codes.get(10)),
        "y": _float(codes.get(20)),
        "text": codes.get(1, ""),
        "height": _float(codes.get(40)),
        "rotation": _float(codes.get(50)),
    }


def _parse_line(codes: dict[int, str]) -> dict[str, Any]:
    return {
        "type": "LINE",
        "layer": codes.get(8, "0"),
        "x1": _float(codes.get(10)),
        "y1": _float(codes.get(20)),
        "x2": _float(codes.get(11)),
        "y2": _float(codes.get(21)),
    }


def _parse_insert(codes: dict[int, str]) -> dict[str, Any]:
    return {
        "block_name": codes.get(2, ""),
        "x": _float(codes.get(10)),
        "y": _float(codes.get(20)),
        "sx": _float(codes.get(41), 1.0),
        "sy": _float(codes.get(42), 1.0),
        "rotation": _float(codes.get(50)),
    }


# ---------------------------------------------------------------------------
# Block-level entity stream parser
# ---------------------------------------------------------------------------

def _parse_entities_stream(
    tokens: list[tuple[int, str]],
    start: int,
    stop_marker: str | None = None,
) -> tuple[list[dict], list[dict], int]:
    """Parse a stream of entities from *start*.

    Stops when it hits (0, stop_marker), EOF, or (0, "BLOCK") for the
    next block.

    Returns (entities, inserts, next_index).
    """
    entities: list[dict] = []
    inserts: list[dict] = []
    idx = start
    n = len(tokens)

    while idx < n:
        c, v = tokens[idx]
        if c == 0:
            if stop_marker and v == stop_marker:
                idx += 1
                # skip trailing codes of ENDBLK
                while idx < n and tokens[idx][0] != 0:
                    idx += 1
                break
            if v == "BLOCK":
                # next block boundary — don't consume
                break
            if v == "POLYLINE":
                idx += 1
                ent, idx = _parse_polyline(tokens, idx)
                entities.append(ent)
                continue
            if v in ("POINT", "TEXT", "LINE", "INSERT"):
                etype = v
                idx += 1
                codes, idx = _collect_entity(tokens, idx)
                if etype == "POINT":
                    entities.append(_parse_point(codes))
                elif etype == "TEXT":
                    entities.append(_parse_text(codes))
                elif etype == "LINE":
                    entities.append(_parse_line(codes))
                elif etype == "INSERT":
                    inserts.append(_parse_insert(codes))
                continue
            # Unknown entity — skip
            idx += 1
            while idx < n and tokens[idx][0] != 0:
                idx += 1
            continue
        idx += 1

    return entities, inserts, idx


# ---------------------------------------------------------------------------
# HEADER parsing
# ---------------------------------------------------------------------------

def _parse_header(tokens: list[tuple[int, str]]) -> dict[str, Any]:
    header: dict[str, Any] = {"units": "mm", "insunits": 4}
    for i, (c, v) in enumerate(tokens):
        if c == 9:
            if v == "$INSUNITS" and i + 1 < len(tokens):
                header["insunits"] = _int(tokens[i + 1][1], 4)
            elif v == "$LUNITS" and i + 1 < len(tokens):
                lu = _int(tokens[i + 1][1], 2)
                unit_map = {1: "scientific", 2: "decimal", 3: "engineering", 4: "architectural"}
                header["units"] = unit_map.get(lu, "decimal")
    return header


# ---------------------------------------------------------------------------
# BLOCKS parsing
# ---------------------------------------------------------------------------

def _parse_blocks(tokens: list[tuple[int, str]]) -> list[dict]:
    blocks: list[dict] = []
    idx = 0
    n = len(tokens)

    while idx < n:
        c, v = tokens[idx]
        if c == 0 and v == "BLOCK":
            idx += 1
            # Collect BLOCK header codes
            block_codes: dict[int, str] = {}
            while idx < n and tokens[idx][0] != 0:
                block_codes[tokens[idx][0]] = tokens[idx][1]
                idx += 1
            block_name = block_codes.get(2, "UNNAMED")

            entities, _, idx = _parse_entities_stream(tokens, idx, stop_marker="ENDBLK")
            blocks.append({"name": block_name, "entities": entities})
        else:
            idx += 1

    return blocks


# ---------------------------------------------------------------------------
# ENTITIES section parsing (INSERT references)
# ---------------------------------------------------------------------------

def _parse_entities_section(tokens: list[tuple[int, str]]) -> tuple[list[dict], list[dict]]:
    entities, inserts, _ = _parse_entities_stream(tokens, 0)
    return entities, inserts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DXFParser:
    """Parse Gemini CAD Systems DXF files into a structured dict."""

    def parse(self, filepath: str) -> dict[str, Any]:
        """Parse a DXF file and return structured data.

        Returns::

            {
                'header': {'units': str, 'insunits': int},
                'blocks': [{'name': str, 'entities': [...]}, ...],
                'inserts': [{'block_name': str, 'x': float, ...}, ...],
            }
        """
        tokens = _tokenize(filepath)
        sections = _split_sections(tokens)

        header = _parse_header(sections.get("HEADER", []))
        blocks = _parse_blocks(sections.get("BLOCKS", []))

        ent_section = sections.get("ENTITIES", [])
        top_entities, inserts = _parse_entities_section(ent_section)

        # Merge any top-level entities (rare but possible) into a synthetic block
        if top_entities:
            blocks.append({"name": "__ENTITIES__", "entities": top_entities})

        return {
            "header": header,
            "blocks": blocks,
            "inserts": inserts,
        }
