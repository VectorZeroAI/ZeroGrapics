#ZeroFiller.py
"""
ZeroFiller.py
- Reads LLM-generated .txt or .json that describes points, lines and shapes.
- Parses the input (JSON preferred) or a legacy brace-based text format.
- Validates fields and inserts rows into the SQLite DB produced by ZeroInit.py.
Schema expected (graphics.db created by ZeroInit.py):
  points(coordinates TEXT, uuid TEXT PRIMARY KEY, connected_points TEXT, movements TEXT)
  lines(uuid TEXT PRIMARY KEY, endpoints TEXT, pull_point TEXT, pull_power REAL, movements TEXT)
  shapes(uuid TEXT PRIMARY KEY, point_uuids TEXT, line_uuids TEXT, color TEXT, movements TEXT)  # FIXED: Added uuid
Notes:
 - Movements / changes are saved to the DB as JSON strings (arrays) when possible.
 - Coordinates/colors are saved as JSON arrays [x,y,z].
 - For legacy semicolon formats the parser attempts to convert to normalized JSON storage.
"""
import argparse
import sqlite3
import json
import os
import re
import sys
import uuid
from typing import List, Tuple, Any
DB_FILE = "graphics.db"

# -----------------------
# Parsing helpers (lightweight, similar to interpreter)
# -----------------------
def try_load_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_vector3_from_any(s: Any) -> Tuple[float, float, float]:
    """
    Accepts: JSON list/tuple or string like "[x,y,z]" or "x;y;z" or "(x;y;z)" or "x,y,z"
    Returns tuple (x,y,z) or raises ValueError.
    """
    if s is None:
        raise ValueError("None vector")
    if isinstance(s, (list, tuple)) and len(s) >= 3:
        return float(s[0]), float(s[1]), float(s[2])
    if isinstance(s, (int, float)):
        # scalar -> replicate? not allowed
        raise ValueError("Scalar cannot be vector3")
    # string fallback
    txt = str(s).strip()
    j = try_load_json(txt)
    if j is not None and isinstance(j, (list, tuple)) and len(j) >= 3:
        return float(j[0]), float(j[1]), float(j[2])
    # remove parentheses/brackets
    for ch in '()[]{}':
        txt = txt.replace(ch, '')
    # semicolon or comma
    for sep in (';', ','):
        if sep in txt:
            parts = [p.strip() for p in txt.split(sep) if p.strip() != '']
            if len(parts) >= 3:
                return float(parts[0]), float(parts[1]), float(parts[2])
    # last resort
    raise ValueError(f"Can't parse vector3 from: {s}")

def normalize_vector3_json(s: Any) -> str:
    v = parse_vector3_from_any(s)
    return json.dumps([float(v[0]), float(v[1]), float(v[2])])

def parse_uuid_list_from_any(s: Any) -> List[str]:
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [str(x) for x in s]
    txt = str(s).strip()
    j = try_load_json(txt)
    if j is not None and isinstance(j, (list, tuple)):
        return [str(x) for x in j]
    for ch in '()[]{}':
        txt = txt.replace(ch, '')
    if ';' in txt:
        parts = [p.strip() for p in txt.split(';') if p.strip() != '']
        return parts
    if ',' in txt:
        parts = [p.strip() for p in txt.split(',') if p.strip() != '']
        return parts
    if txt == '':
        return []
    return [txt]

def normalize_uuid_list_json(s: Any) -> str:
    arr = parse_uuid_list_from_any(s)
    return json.dumps(arr)

def parse_movements_from_any(s: Any) -> List[List[Any]]:
    """
    Convert various movement encodings into a canonical list:
      [ [start_ms, duration_ms, [x,y,z]], ... ]
    Accepts JSON lists or semicolon text like "(100;500;[1;2;3]);(600;200;[2;3;4])"
    """
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        out = []
        for item in s:
            # allow item to be [start,dur,[x,y,z]] or dict
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                start = int(item[0]); dur = int(item[1]); tgt = item[2]
                # parse target vector
                tx,ty,tz = parse_vector3_from_any(tgt)
                out.append([start, dur, [tx,ty,tz]])
            elif isinstance(item, dict):
                # try keys start,duration,target
                start = int(item.get("start", item.get("wait", 0)))
                dur = int(item.get("duration", item.get("dur", 0)))
                tgt = item.get("target", item.get("coords", None))
                tx,ty,tz = parse_vector3_from_any(tgt)
                out.append([start, dur, [tx,ty,tz]])
        return out
    # string parse: find segments like "(...)" or "[...]" or {...}
    txt = str(s).strip()
    if txt == "":
        return []
    segments = re.findall(r'[\(\[\{]([^}\]\)]*)[\)\]\}]', txt)
    out = []
    for seg in segments:
        # seg e.g. "100;500;[1;2;3]" or "100,500,[1,2,3]"
        # split on semicolon or comma but keep bracketed part intact
        # naive approach: split by ';' first
        parts = [p.strip() for p in seg.split(';') if p.strip()!='']
        if len(parts) >= 3:
            try:
                start = int(float(parts[0])); dur = int(float(parts[1]))
                tgt_raw = parts[2]
                try:
                    tgt = parse_vector3_from_any(tgt_raw)
                except ValueError:
                    # maybe parts 2,3,4 are the coords
                    if len(parts) >= 5:
                        tgt = (float(parts[2]), float(parts[3]), float(parts[4]))
                    else:
                        raise
                out.append([start, dur, [tgt[0], tgt[1], tgt[2]]])
            except Exception:
                continue
    return out

def normalize_movements_json(s: Any) -> str:
    arr = parse_movements_from_any(s)
    return json.dumps(arr)

def parse_line_changes_from_any(s: Any) -> List[List[Any]]:
    """
    Convert line change encodings into:
      [ [start_ms, duration_ms, [x,y,z,power]], ... ]
    """
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        out = []
        for item in s:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                start = int(item[0]); dur = int(item[1]); third = item[2]
                if isinstance(third, (list, tuple)) and len(third) >= 4:
                    x,y,z,p = float(third[0]), float(third[1]), float(third[2]), float(third[3])
                elif len(item) >= 6:
                    x,y,z,p = float(item[2]), float(item[3]), float(item[4]), float(item[5])
                else:
                    continue
                out.append([start, dur, [x,y,z,p]])
        return out
    txt = str(s).strip()
    if txt == "":
        return []
    # find bracketed segments
    segments = re.findall(r'[\(\[\{]([^}\]\)]*)[\)\]\}]', txt)
    out = []
    for seg in segments:
        parts = [p.strip() for p in seg.split(';') if p.strip()!='']
        if len(parts) >= 6:
            try:
                start = int(float(parts[0])); dur = int(float(parts[1]))
                x = float(parts[2]); y = float(parts[3]); z = float(parts[4]); p = float(parts[5])
                out.append([start, dur, [x,y,z,p]])
            except Exception:
                continue
    return out

def normalize_line_changes_json(s: Any) -> str:
    arr = parse_line_changes_from_any(s)
    return json.dumps(arr)

def parse_color_changes_from_any(s: Any) -> List[List[Any]]:
    # format [ [start,dur,[r,g,b]], ... ]
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        out=[]
        for item in s:
            if isinstance(item,(list,tuple)) and len(item)>=3:
                start=int(item[0]); dur=int(item[1]); col=item[2]
                r,g,b = parse_vector3_from_any(col)
                out.append([start,dur,[r,g,b]])
        return out
    txt=str(s).strip()
    if txt=="":
        return []
    segments = re.findall(r'[\(\[\{]([^}\]\)]*)[\)\]\}]', txt)
    out=[]
    for seg in segments:
        parts=[p.strip() for p in seg.split(';') if p.strip()!='']
        if len(parts)>=5:
            try:
                start=int(float(parts[0])); dur=int(float(parts[1]))
                r=float(parts[2]); g=float(parts[3]); b=float(parts[4])
                out.append([start,dur,[r,g,b]])
            except:
                continue
    return out

def normalize_color_changes_json(s: Any) -> str:
    arr = parse_color_changes_from_any(s)
    return json.dumps(arr)

# -----------------------
# High-level insertion (FIXED: Added uuid to shapes)
# -----------------------
def insert_point(conn: sqlite3.Connection, coords_json: str, uuid_str: str, connected_json: str, movements_json: str):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO points (coordinates, uuid, connected_points, movements) VALUES (?, ?, ?, ?)",
                (coords_json, uuid_str, connected_json, movements_json))
    conn.commit()

def insert_line(conn: sqlite3.Connection, uuid_str: str, endpoints_json: str, pull_json: str, power_float: float, changes_json: str):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO lines (uuid, endpoints, pull_point, pull_power, movements) VALUES (?, ?, ?, ?, ?)",
                (uuid_str, endpoints_json, pull_json, power_float, changes_json))
    conn.commit()

# FIXED: Added uuid parameter to shape insertion
def insert_shape(conn: sqlite3.Connection, uuid_str: str, points_json: str, lines_json: str, color_json: str, color_changes_json: str):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO shapes (uuid, point_uuids, line_uuids, color, movements) VALUES (?, ?, ?, ?, ?)",
                (uuid_str, points_json, lines_json, color_json, color_changes_json))
    conn.commit()

# -----------------------
# Legacy text parsing helpers
# -----------------------
def extract_table_block(text: str, table_marker: str) -> str:
    """
    Return the block of text after the first occurrence of table_marker
    (e.g. "TABLE1") up to the next TABLE[1-3] marker or EOF.

    Examples:
      extract_table_block(raw_text, "TABLE1")
    Will return the text immediately after "TABLE1" (case-insensitive)
    until the next "TABLE1/2/3" header or end of file.
    """
    if not text or not table_marker:
        return ""

    # Find the marker (case-insensitive)
    m = re.search(re.escape(table_marker), text, re.IGNORECASE)
    if not m:
        return ""

    start = m.end()
    # Find next TABLE1|TABLE2|TABLE3 header after the marker
    nxt = re.search(r'\bTABLE[123]\b', text[start:], re.IGNORECASE)
    if nxt:
        end = start + nxt.start()
    else:
        end = len(text)

    return text[start:end].strip()

def parse_legacy_rows(block: str) -> List[List[str]]:
    """
    For a table block, find rows. Each row may be a line or a semicolon separated group.
    We extract all {...} bracket contents per row and return list-of-lists (fields).
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()!='']
    rows = []
    for ln in lines:
        # find all {...} groups; if none, try split by 'ROW' or by ';;'
        groups = re.findall(r'\{([^}]*)\}', ln)
        if not groups:
            # fallback: split by '|' or ';;'
            parts = re.split(r'\s*\|\s*|\s*;;\s*', ln)
            parts = [p.strip() for p in parts if p.strip()!='']
            rows.append(parts)
        else:
            rows.append(groups)
    return rows

# -----------------------
# Main parser & writer
# -----------------------
def process_json_input(conn: sqlite3.Connection, data: dict):
    # Expected keys: points, lines, shapes
    inserted = {"points":0,"lines":0,"shapes":0}
    # POINTS
    pts = data.get("points", []) or []
    for p in pts:
        try:
            coords = normalize_vector3_json(p.get("coords", p.get("coordinates", None)))
        except Exception as e:
            print("Skipping point with bad coords:", p, "err:", e)
            continue
        uuid_str = str(p.get("uuid") or p.get("id") or str(uuid.uuid4()))
        # connections
        connected_json = normalize_uuid_list_json(p.get("connections", p.get("connected_points", [])))
        movements_json = normalize_movements_json(p.get("movements", p.get("changes", [])))
        insert_point(conn, coords, uuid_str, connected_json, movements_json)
        inserted["points"] += 1
    # LINES
    lns = data.get("lines", []) or []
    for l in lns:
        try:
            uuid_str = str(l.get("uuid") or l.get("id") or str(uuid.uuid4()))
            endpoints = normalize_uuid_list_json(l.get("endpoints", l.get("endpoints_list", [])))
            pull_json = normalize_vector3_json(l.get("pull_point", l.get("pull_coords", [0,0,0])))
            power = float(l.get("pull_power", l.get("power", 1.0) or 1.0))
            changes_json = normalize_line_changes_json(l.get("changes", l.get("movements", [])))
            insert_line(conn, uuid_str, endpoints, pull_json, power, changes_json)
            inserted["lines"] += 1
        except Exception as e:
            print("Skipping line due to error:", e, l)
            continue
    # SHAPES (FIXED: Added uuid handling)
    shs = data.get("shapes", []) or []
    for s in shs:
        try:
            uuid_str = str(s.get("uuid") or str(uuid.uuid4()))
            points_json = normalize_uuid_list_json(s.get("points", s.get("point_uuids", [])))
            lines_json = normalize_uuid_list_json(s.get("lines", s.get("line_uuids", [])))
            color_json = normalize_vector3_json(s.get("color", s.get("fill_color", [1.0,1.0,1.0])))
            color_changes_json = normalize_color_changes_json(s.get("color_changes", s.get("movements", [])))
            insert_shape(conn, uuid_str, points_json, lines_json, color_json, color_changes_json)
            inserted["shapes"] += 1
        except Exception as e:
            print("Skipping shape due to error:", e, s)
            continue
    return inserted

def process_legacy_text(conn: sqlite3.Connection, raw_text: str):
    inserted = {"points":0,"lines":0,"shapes":0}
    # Extract blocks
    tbl1 = extract_table_block(raw_text, "TABLE1")
    tbl2 = extract_table_block(raw_text, "TABLE2")
    tbl3 = extract_table_block(raw_text, "TABLE3")
    # TABLE1 -> points
    rows1 = parse_legacy_rows(tbl1)
    for fields in rows1:
        # expected order: coords, uuid, connected_points, movements
        try:
            coords_raw = fields[0] if len(fields) >= 1 else ""
            uuid_raw = fields[1] if len(fields) >= 2 else ""
            conn_raw = fields[2] if len(fields) >= 3 else ""
            mov_raw = fields[3] if len(fields) >= 4 else ""
            coords = normalize_vector3_json(coords_raw)
            uuid_str = str(uuid_raw) if uuid_raw.strip()!='' else str(uuid.uuid4())
            connected_json = normalize_uuid_list_json(conn_raw)
            movements_json = normalize_movements_json(mov_raw)
            insert_point(conn, coords, uuid_str, connected_json, movements_json)
            inserted["points"] += 1
        except Exception as e:
            print("Skipping legacy point row - error:", e, fields)
            continue
    # TABLE2 -> lines
    rows2 = parse_legacy_rows(tbl2)
    for fields in rows2:
        # expected: uuid, endpoints, pull_coords, pull_power, changes
        try:
            uuid_raw = fields[0] if len(fields) >= 1 else ""
            endpoints_raw = fields[1] if len(fields) >= 2 else ""
            pull_raw = fields[2] if len(fields) >= 3 else ""
            power_raw = fields[3] if len(fields) >= 4 else "1.0"
            changes_raw = fields[4] if len(fields) >= 5 else ""
            uuid_str = str(uuid_raw) if str(uuid_raw).strip()!='' else str(uuid.uuid4())
            endpoints_json = normalize_uuid_list_json(endpoints_raw)
            pull_json = normalize_vector3_json(pull_raw)
            power = float(power_raw) if str(power_raw).strip()!='' else 1.0
            changes_json = normalize_line_changes_json(changes_raw)
            insert_line(conn, uuid_str, endpoints_json, pull_json, power, changes_json)
            inserted["lines"] += 1
        except Exception as e:
            print("Skipping legacy line row - error:", e, fields)
            continue
    # TABLE3 -> shapes (FIXED: Added uuid handling)
    rows3 = parse_legacy_rows(tbl3)
    for fields in rows3:
        # expected: uuid, point_uuids, line_uuids, color, movements
        try:
            uuid_raw = fields[0] if len(fields) >= 1 else ""
            pts_raw = fields[1] if len(fields) >= 2 else ""
            lns_raw = fields[2] if len(fields) >= 3 else ""
            color_raw = fields[3] if len(fields) >= 4 else "[1,1,1]"
            mov_raw = fields[4] if len(fields) >= 5 else ""
            uuid_str = str(uuid_raw) if str(uuid_raw).strip()!='' else str(uuid.uuid4())
            pts_json = normalize_uuid_list_json(pts_raw)
            lns_json = normalize_uuid_list_json(lns_raw)
            color_json = normalize_vector3_json(color_raw)
            color_changes_json = normalize_color_changes_json(mov_raw)
            insert_shape(conn, uuid_str, pts_json, lns_json, color_json, color_changes_json)
            inserted["shapes"] += 1
        except Exception as e:
            print("Skipping legacy shape row - error:", e, fields)
            continue
    return inserted

# -----------------------
# CLI & main
# -----------------------
def clear_tables(conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("DELETE FROM points;")
    c.execute("DELETE FROM lines;")
    c.execute("DELETE FROM shapes;")
    conn.commit()

def main():
    ap = argparse.ArgumentParser(description="ZeroFiller - fill graphics.db from LLM output (JSON or legacy text).")
    ap.add_argument("input", nargs="?", help="Input file (JSON or text). Use --stdin to read from stdin.")
    ap.add_argument("--stdin", action="store_true", help="Read input from STDIN")
    ap.add_argument("--db", default=DB_FILE, help="Path to SQLite DB (default graphics.db)")
    ap.add_argument("--clear", action="store_true", help="Clear tables before inserting")
    ap.add_argument("--seed-example", action="store_true", help="Insert a tiny example after creating DB (no-op here)")
    args = ap.parse_args()
    if not os.path.exists(args.db):
        print("DB not found:", args.db)
        print("Run ZeroInit.py to create the DB first.")
        sys.exit(1)
    if args.stdin:
        raw = sys.stdin.read()
    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        print("Provide an input file or use --stdin")
        ap.print_help()
        return
    # Try parse JSON first
    j = try_load_json(raw)
    conn = sqlite3.connect(args.db)
    if args.clear:
        clear_tables(conn)
        print("Cleared existing tables.")
    inserted = {"points":0,"lines":0,"shapes":0}
    if j is not None and isinstance(j, dict):
        print("Detected JSON input. Processing...")
        inserted = process_json_input(conn, j)
    else:
        print("Treating input as legacy text format. Parsing...")
        inserted = process_legacy_text(conn, raw)
    conn.close()
    print("Insertion summary:", inserted)
    print("Done.")

if __name__ == "__main__":
    main()