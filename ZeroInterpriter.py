"""
ZeroInterpreter.py
- Connects to an SQLite DB with the tables:
  points(uuid TEXT, coords TEXT, connections TEXT, movements TEXT)
  lines(uuid TEXT, endpoints TEXT, pull_coords TEXT, pull_power REAL, changes TEXT)
  shapes(uuid TEXT, point_list TEXT, line_list TEXT, fill_color TEXT, color_changes TEXT)

- Supports two field formats:
  * JSON-like arrays:  [x,y,z]  or  [[...],...]
  * semicolon-separated: "x;y;z"  or  "(x;y;[a;b;c]);..." etc.

- Public method: read_full(tick_in_ms) -> frame dict suitable for rendering.
"""

import sqlite3
import json
import math
from typing import List, Tuple, Dict, Any

# ----------------------
# Utility / parsing
# ----------------------
def try_parse_json(text: str):
    if text is None: 
        return None
    text = text.strip()
    if text == "":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

def parse_vector3(text: str) -> Tuple[float, float, float]:
    """
    Accepts:
      - JSON array: [x, y, z]
      - Semicolon: "x;y;z"
      - Parenthesized: "(x;y;z)"
    Raises ValueError if cannot parse.
    """
    if text is None:
        raise ValueError("None vector")
    txt = text.strip()
    j = try_parsejson_or_none(txt := txt)
    if j is not None and isinstance(j, (list, tuple)) and len(j) >= 3:
        return float(j[0]), float(j[1]), float(j[2])
    # fallback semicolon
    # remove parentheses/brackets
    for ch in '()[]':
        txt = txt.replace(ch, '')
    parts = [p.strip() for p in txt.split(';') if p.strip() != '']
    if len(parts) >= 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    # try comma-separated
    parts = [p.strip() for p in txt.split(',') if p.strip() != '']
    if len(parts) >= 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    raise ValueError(f"Can't parse vector3 from: {text}")

def parse_vectorN(text: str) -> List[float]:
    # Generic: try JSON then semicolon then comma
    if text is None:
        return []
    j = try_parsejson_or_none(text)
    if j is not None and isinstance(j, (list, tuple)):
        return [float(x) for x in j]
    s = text.strip()
    for ch in '()[]':
        s = s.replace(ch, '')
    if ';' in s:
        parts = [p.strip() for p in s.split(';') if p.strip() != '']
        return [float(x) for x in parts]
    if ',' in s:
        return [float(x) for x in s.split(',') if x.strip()!='']
    # single number
    try:
        return [float(s)]
    except:
        return []

def parse_uuid_list(text: str) -> List[str]:
    if text is None:
        return []
    j = try_parsejson_or_none(text)
    if j is not None and isinstance(j, (list, tuple)):
        return [str(x) for x in j]
    s = text.strip()
    for ch in '()[]':
        s = s.replace(ch, '')
    if ';' in s:
        return [p.strip() for p in s.split(';') if p.strip()!='']
    if ',' in s:
        return [p.strip() for p in s.split(',') if p.strip()!='']
    if s == '':
        return []
    return [s]

def try_parsejson_or_none(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

# ----------------------
# Interpolation helpers
# ----------------------
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def lerp_vec(a: Tuple[float,float,float], b: Tuple[float,float,float], t: float) -> Tuple[float,float,float]:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))

def clamp(t: float, lo: float=0.0, hi: float=1.0):
    return max(lo, min(hi, t))

def vec_distance(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

# ----------------------
# Movement parsing models
# ----------------------
# Movement format (per user): (wait; duration; [x;y;z])
# We'll support JSON arrays of [wait, duration, [x,y,z]] or semicolon text.

def parse_movements_field(text: str) -> List[Tuple[int,int,Tuple[float,float,float]]]:
    """
    Returns list of (start_ms, duration_ms, target_xyz)
    """
    if text is None:
        return []
    j = try_parsejson_or_none(text)
    out = []
    if j is not None and isinstance(j, (list, tuple)):
        # each element could be list/tuple
        for item in j:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    start = int(item[0])
                    dur = int(item[1])
                    tgt = item[2]
                    if isinstance(tgt, (list, tuple)):
                        tx,ty,tz = float(tgt[0]), float(tgt[1]), float(tgt[2])
                    else:
                        tx,ty,tz = parse_vector3(str(tgt))
                    out.append((start, dur, (tx,ty,tz)))
            except Exception:
                continue
        return sorted(out, key=lambda x: x[0])
    # Otherwise parse semicolon groups like "(x;y;[a;b;c]);..."
    s = text.strip()
    if s == "":
        return []
    # split on '),' or ');' heuristically
    segments = []
    cur = ""
    depth = 0
    for ch in s:
        cur += ch
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if depth == 0 and (ch == ')' or ch == ']'):
            segments.append(cur.strip())
            cur = ""
    if cur.strip() != "":
        segments.append(cur.strip())
    if not segments:
        # fallback split by semicolon triple groups
        parts = [p for p in s.split(')') if p.strip()!='']
        segments = parts
    for seg in segments:
        # remove surrounding parentheses/brackets
        seg_clean = seg.strip()
        for ch in '()[]':
            seg_clean = seg_clean.replace(ch, '')
        # now seg_clean like "wait;duration; a;b;c" or "wait;duration;[a;b;c]"
        parts = [p.strip() for p in seg_clean.split(';') if p.strip()!='']
        if len(parts) >= 5:
            try:
                start = int(parts[0])
                dur = int(parts[1])
                tx = float(parts[2]); ty = float(parts[3]); tz = float(parts[4])
                out.append((start, dur, (tx,ty,tz)))
            except Exception:
                continue
    return sorted(out, key=lambda x: x[0])

# ----------------------
# Line change parsing
# ----------------------
# Line changes format: {(x;y;[a;b;c;d]);...}
# where a,b,c are new pull coords and d is new pull power
def parse_line_changes_field(text: str) -> List[Tuple[int,int,Tuple[float,float,float], float]]:
    if text is None:
        return []
    j = try_parsejson_or_none(text)
    out = []
    if j is not None and isinstance(j, (list, tuple)):
        for item in j:
            # expect [start,duration,[x,y,z,d]] or [start,duration,x,y,z,d]
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    start = int(item[0]); dur = int(item[1])
                    third = item[2]
                    if isinstance(third, (list, tuple)) and len(third) >= 4:
                        x,y,z,p = float(third[0]), float(third[1]), float(third[2]), float(third[3])
                    else:
                        # maybe flattened
                        x,y,z,p = float(item[2]), float(item[3]), float(item[4]), float(item[5])
                    out.append((start,dur,(x,y,z), float(p)))
            except Exception:
                continue
        return sorted(out, key=lambda x:x[0])
    # fallback semicolon parsing similar to movements
    s = text.strip()
    if s == "":
        return []
    # naive split by segments
    segments = []
    cur = ""; depth = 0
    for ch in s:
        cur += ch
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if depth == 0 and (ch == ')' or ch == ']'):
            segments.append(cur.strip()); cur = ""
    if cur.strip() != "":
        segments.append(cur.strip())
    for seg in segments:
        seg_clean = seg
        for ch in '()[]':
            seg_clean = seg_clean.replace(ch, '')
        parts = [p.strip() for p in seg_clean.split(';') if p.strip()!='']
        if len(parts) >= 6:
            try:
                start = int(parts[0]); dur = int(parts[1])
                x = float(parts[2]); y = float(parts[3]); z = float(parts[4]); p = float(parts[5])
                out.append((start,dur,(x,y,z),p))
            except Exception:
                continue
    return sorted(out, key=lambda x:x[0])

# ----------------------
# Color change parsing
# ----------------------
# Colors: { (start;duration;[r;g;b]); ... }
def parse_color_changes(text: str) -> List[Tuple[int,int,Tuple[float,float,float]]]:
    if text is None:
        return []
    j = try_parsejson_or_none(text)
    out = []
    if j is not None and isinstance(j, (list, tuple)):
        for item in j:
            try:
                start = int(item[0]); dur = int(item[1])
                third = item[2]
                if isinstance(third, (list,tuple)) and len(third) >= 3:
                    r,g,b = float(third[0]), float(third[1]), float(third[2])
                else:
                    r,g,b = parse_vectorN(str(third))[:3]
                out.append((start,dur,(r,g,b)))
            except Exception:
                continue
        return sorted(out, key=lambda x:x[0])
    # fallback simple parse like earlier
    s = text.strip()
    if s == "":
        return []
    segments = []
    cur = ""; depth=0
    for ch in s:
        cur += ch
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if depth==0 and (ch==')' or ch==']'):
            segments.append(cur.strip()); cur=""
    if cur.strip()!="":
        segments.append(cur.strip())
    for seg in segments:
        seg_clean = seg
        for ch in '()[]':
            seg_clean = seg_clean.replace(ch, '')
        parts = [p.strip() for p in seg_clean.split(';') if p.strip()!='']
        if len(parts) >= 5:
            try:
                start = int(parts[0]); dur = int(parts[1])
                r = float(parts[2]); g = float(parts[3]); b = float(parts[4])
                out.append((start,dur,(r,g,b)))
            except:
                continue
    return sorted(out, key=lambda x:x[0])

# ----------------------
# Bézier sampling
# ----------------------
def sample_quadratic_bezier(p0, pc, p1, n_samples=16):
    samples = []
    if n_samples < 2:
        n_samples = 2
    for i in range(n_samples+1):
        t = i / n_samples
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * pc[0] + t**2 * p1[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * pc[1] + t**2 * p1[1]
        z = (1-t)**2 * p0[2] + 2*(1-t)*t * pc[2] + t**2 * p1[2]
        samples.append((x,y,z))
    return samples

# choose sample count by approximate curve length
def sample_count_for_curve(p0, pc, p1, min_samples=8):
    approx_len = vec_distance(p0, pc) + vec_distance(pc, p1)
    # 0.05 samples per unit length (tweakable)
    n = max(min_samples, int(approx_len * 0.5))
    return n

# ----------------------
# Main interpreter class
# ----------------------
class ZeroInterpreter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.points = {}   # uuid -> { base: (x,y,z), movements: [...] }
        self.lines = {}    # uuid -> { endpoints: [u1,u2], pull_base: (x,y,z), pull_power_base, changes: [...] }
        self.shapes = {}   # uuid -> { points: [...], lines: [...], fill_base: (r,g,b), color_changes: [...] }
        self._load_all()

    def _load_all(self):
        cur = self.conn.cursor()
        # Points
        try:
            rows = cur.execute("SELECT coords, uuid, connections, movements FROM points").fetchall()
        except Exception:
            rows = []
        idx = 0
        for r in rows:
            try:
                uuid = str(r["uuid"])
                base = parse_vector3(r["coords"])
            except Exception:
                # skip malformed
                continue
            conns = parse_uuid_list(r["connections"]) if "connections" in r.keys() else []
            moves = parse_movements_field(r["movements"]) if "movements" in r.keys() else []
            self.points[uuid] = {"base": base, "connections": conns, "movements": moves}
            idx += 1

        # Lines
        try:
            rows = cur.execute("SELECT uuid, endpoints, pull_coords, pull_power, changes FROM lines").fetchall()
        except Exception:
            rows = []
        for r in rows:
            try:
                uuid = str(r["uuid"])
                endpoints = parse_uuid_list(r["endpoints"])
                pull_base = parse_vector3(r["pull_coords"])
                power_base = float(r["pull_power"]) if r["pull_power"] is not None else 1.0
                changes = parse_line_changes_field(r["changes"]) if "changes" in r.keys() else []
                self.lines[uuid] = {"endpoints": endpoints, "pull_base": pull_base, "power_base": power_base, "changes": changes}
            except Exception:
                continue

        # Shapes
        try:
            rows = cur.execute("SELECT uuid, point_list, line_list, fill_color, color_changes FROM shapes").fetchall()
        except Exception:
            rows = []
        for r in rows:
            try:
                uuid = str(r["uuid"])
                pts = parse_uuid_list(r["point_list"])
                lns = parse_uuid_list(r["line_list"])
                fill = tuple(parse_vectorN(r["fill_color"])[:3]) if r["fill_color"] is not None else (1.0,1.0,1.0)
                color_changes = parse_color_changes(r["color_changes"]) if "color_changes" in r.keys() else []
                self.shapes[uuid] = {"points": pts, "lines": lns, "fill_base": fill, "color_changes": color_changes}
            except Exception:
                continue

    # ----------------------
    # compute point position at tick
    # ----------------------
    def point_position_at(self, uuid: str, tick_ms: int) -> Tuple[float,float,float]:
        rec = self.points.get(uuid)
        if rec is None:
            raise KeyError(f"Point {uuid} not found")
        base = rec["base"]
        movements = rec["movements"]  # list of (start,dur,target)
        # start from base, process movements in order
        current = base
        for m in movements:
            start, dur, target = m
            if tick_ms < start:
                # haven't reached this movement yet -> still at current
                return current
            end = start + dur
            if dur <= 0:
                # instant jump at start
                current = target
                continue
            if tick_ms >= end:
                # finished this movement -> update current to target
                current = target
                continue
            # we're in this movement
            # starting position for this movement is 'current'
            progress = (tick_ms - start) / dur
            progress = clamp(progress, 0.0, 1.0)
            return lerp_vec(current, target, progress)
        # no more movements -> last known position
        return current

    # ----------------------
    # compute pull point and power at tick for a line
    # ----------------------
    def line_pull_at(self, line_uuid: str, tick_ms: int) -> Tuple[Tuple[float,float,float], float]:
        rec = self.lines.get(line_uuid)
        if rec is None:
            raise KeyError(f"Line {line_uuid} not found")
        base = rec["pull_base"]
        base_power = rec["power_base"]
        changes = rec["changes"]  # list of (start,dur,(x,y,z), power)
        current_pos = base
        current_power = base_power
        for ch in changes:
            start, dur, new_coords, new_power = ch
            if tick_ms < start:
                return current_pos, current_power
            end = start + dur
            if dur <= 0:
                current_pos = new_coords
                current_power = new_power
                continue
            if tick_ms >= end:
                current_pos = new_coords
                current_power = new_power
                continue
            # in-progress
            progress = clamp((tick_ms - start) / dur)
            curp = lerp_vec(current_pos, new_coords, progress)
            curpow = lerp(current_power, new_power, progress)
            return curp, curpow
        return current_pos, current_power

    # ----------------------
    # compute shape color at tick
    # ----------------------
    def shape_color_at(self, shape_uuid: str, tick_ms: int) -> Tuple[float,float,float]:
        rec = self.shapes.get(shape_uuid)
        if rec is None:
            raise KeyError(f"Shape {shape_uuid} not found")
        base = rec["fill_base"]
        changes = rec["color_changes"]
        current = base
        for ch in changes:
            start,dur,newcol = ch
            if tick_ms < start:
                return current
            end = start + dur
            if dur <= 0:
                current = newcol
                continue
            if tick_ms >= end:
                current = newcol
                continue
            # in-progress
            progress = clamp((tick_ms - start)/dur)
            return (lerp(current[0], newcol[0], progress),
                    lerp(current[1], newcol[1], progress),
                    lerp(current[2], newcol[2], progress))
        return current

    # ----------------------
    # Public: build full frame
    # ----------------------
    def read_full(self, tick_in_ms: int) -> Dict[str, Any]:
        # 1) compute all point positions
        point_positions: Dict[str, Tuple[float,float,float]] = {}
        for uuid in self.points.keys():
            point_positions[uuid] = self.point_position_at(uuid, tick_in_ms)

        # 2) compute lines: for each line, get endpoints positions and pull at tick, sample bezier
        lines_out = []
        for uuid, rec in self.lines.items():
            endpoints = rec["endpoints"]
            if len(endpoints) < 2:
                continue
            u0, u1 = endpoints[0], endpoints[1]
            if u0 not in point_positions or u1 not in point_positions:
                continue
            p0 = point_positions[u0]
            p1 = point_positions[u1]
            pull_point, pull_power = self.line_pull_at(uuid, tick_in_ms)
            # optionally, we could bias pull by pull_power, but quadratic bezier uses pc directly.
            n_samples = sample_count_for_curve(p0, pull_point, p1)
            samples = sample_quadratic_bezier(p0, pull_point, p1, n_samples)
            lines_out.append({
                "uuid": uuid,
                "from_uuid": u0,
                "to_uuid": u1,
                "from": p0,
                "to": p1,
                "pull_point": pull_point,
                "pull_power": pull_power,
                "samples": samples
            })

        # 3) shapes: compute fill colors and triangulate points (fan)
        shapes_out = []
        for uuid, rec in self.shapes.items():
            pts_uuids = rec["points"]
            tri_points = []
            coords_for_fan = []
            for pu in pts_uuids:
                if pu in point_positions:
                    coords_for_fan.append(point_positions[pu])
            # simple fan triangulation if >=3 points
            triangles = []
            if len(coords_for_fan) >= 3:
                a0 = coords_for_fan[0]
                for i in range(1, len(coords_for_fan)-1):
                    triangles.append([a0, coords_for_fan[i], coords_for_fan[i+1]])
            color = self.shape_color_at(uuid, tick_in_ms)
            shapes_out.append({
                "uuid": uuid,
                "triangles": triangles,
                "color": color
            })

        # 4) prepare points list for output
        points_out = [{"uuid": u, "pos": point_positions[u]} for u in point_positions.keys()]

        frame = {
            "timestamp_ms": tick_in_ms,
            "points": points_out,
            "lines": lines_out,
            "shapes": shapes_out
        }
        return frame

# ----------------------
# Example quick test (in-memory DB)
# ----------------------
def _example_in_memory_db():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE points(coords TEXT, uuid TEXT, connections TEXT, movements TEXT)")
    cur.execute("CREATE TABLE lines(uuid TEXT, endpoints TEXT, pull_coords TEXT, pull_power REAL, changes TEXT)")
    cur.execute("CREATE TABLE shapes(uuid TEXT, point_list TEXT, line_list TEXT, fill_color TEXT, color_changes TEXT)")
    # points: A, B
    cur.execute("INSERT INTO points VALUES(?, ?, ?, ?)",
                (json.dumps([0.0,0.0,0.0]), "A", json.dumps(["B"]), json.dumps([[0, 1000, [0.0, 1.0, 0.0]]])))
    cur.execute("INSERT INTO points VALUES(?, ?, ?, ?)",
                (json.dumps([2.0,0.0,0.0]), "B", json.dumps(["A"]), json.dumps([[500, 1000, [2.0, 1.0, 0.0]]])))
    # line AB with pull point above
    cur.execute("INSERT INTO lines VALUES(?, ?, ?, ?, ?)",
                ("L1", json.dumps(["A","B"]), json.dumps([1.0, 1.0, 1.5]), 1.0, json.dumps([[0, 0, [1.0,1.0,1.5, 1.0]]])))
    # shape triangle (A,B plus an extra C)
    cur.execute("INSERT INTO points VALUES(?, ?, ?, ?)",
                (json.dumps([1.0, 0.5, 0.0]), "C", json.dumps([]), None))
    cur.execute("INSERT INTO shapes VALUES(?, ?, ?, ?, ?)",
                ("S1", json.dumps(["A","B","C"]), json.dumps(["L1"]), json.dumps([0.2,0.7,0.3]), None))
    conn.commit()
    return conn

def run_example():
    conn = _example_in_memory_db()
    # save to a temp file for interpreter
    import tempfile
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    path = tf.name
    tf.close()
    conn.backup(sqlite3.connect(path))
    conn.close()
    interp = ZeroInterpreter(path)
    for t in [0, 250, 500, 750, 1000, 1250, 2000]:
        frame = interp.read_full(t)
        print(f"t={t}ms -> {len(frame['points'])} points, {len(frame['lines'])} lines, {len(frame['shapes'])} shapes")
        # print first point pos
        for p in frame['points']:
            print("  point", p['uuid'], "pos", p['pos'])
        print("  line sample count", len(frame['lines'][0]['samples']) if frame['lines'] else 0)
        print("---")

# run_example()  # uncomment to run the demo when executing this file directly

if __name__ == "__main__":
    run_example()