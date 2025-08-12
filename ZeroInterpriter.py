"""
ZeroInterpreter.py
- Connects to an SQLite DB with the tables:
  points(uuid TEXT, coordinates TEXT, connected_points TEXT, movements TEXT)
  lines(uuid TEXT, endpoints TEXT, pull_point TEXT, pull_power REAL, movements TEXT)
  shapes(uuid TEXT, point_uuids TEXT, line_uuids TEXT, color TEXT, movements TEXT)  # FIXED: Added uuid
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
    """Safely parse JSON or return None for invalid input"""
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
    # Try JSON parsing first
    j = try_parse_json(txt)
    if j is not None and isinstance(j, (list, tuple)) and len(j) >= 3:
        return float(j[0]), float(j[1]), float(j[2])
    # Clean up parentheses/brackets for fallback parsing
    for ch in '()[]{}':
        txt = txt.replace(ch, '')
    txt = txt.strip()
    # Try semicolon-separated values
    if ';' in txt:
        parts = [p.strip() for p in txt.split(';') if p.strip() != '']
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
    # Try comma-separated values
    parts = [p.strip() for p in txt.split(',') if p.strip() != '']
    if len(parts) >= 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    raise ValueError(f"Can't parse vector3 from: {text}")

def parse_vectorN(text: str) -> List[float]:
    """Parse any vector format into list of floats"""
    if text is None:
        return []
    # Try JSON parsing
    j = try_parse_json(text)
    if j is not None and isinstance(j, (list, tuple)):
        return [float(x) for x in j]
    # Clean up formatting characters
    s = text.strip()
    for ch in '()[]{}':
        s = s.replace(ch, '')
    s = s.strip()
    # Handle semicolon or comma separated values
    if ';' in s:
        parts = [p.strip() for p in s.split(';') if p.strip() != '']
        return [float(x) for x in parts]
    if ',' in s:
        return [float(x) for x in s.split(',') if x.strip() != '']
    # Single number case
    try:
        return [float(s)]
    except:
        return []

def parse_uuid_list(text: str) -> List[str]:
    """Parse UUID lists from various formats"""
    if text is None:
        return []
    # Try JSON parsing
    j = try_parse_json(text)
    if j is not None and isinstance(j, (list, tuple)):
        return [str(x) for x in j]
    # Clean up formatting characters
    s = text.strip()
    for ch in '()[]{}':
        s = s.replace(ch, '')
    s = s.strip()
    # Split by delimiters
    if ';' in s:
        return [p.strip() for p in s.split(';') if p.strip() != '']
    if ',' in s:
        return [p.strip() for p in s.split(',') if p.strip() != '']
    if s == '':
        return []
    return [s]

# ----------------------
# Interpolation helpers
# ----------------------
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b"""
    return a + (b - a) * t

def lerp_vec(a: Tuple[float,float,float], b: Tuple[float,float,float], t: float) -> Tuple[float,float,float]:
    """Vector linear interpolation"""
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))

def clamp(t: float, lo: float=0.0, hi: float=1.0) -> float:
    """Clamp value between lo and hi"""
    return max(lo, min(hi, t))

def vec_distance(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    """Calculate Euclidean distance between vectors"""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

# ----------------------
# Movement parsing models
# ----------------------
def parse_movements_field(text: str) -> List[Tuple[int,int,Tuple[float,float,float]]]:
    """
    Returns list of (start_ms, duration_ms, target_xyz)
    Handles both JSON arrays and semicolon formats
    """
    if text is None:
        return []
    # Try JSON parsing first
    j = try_parse_json(text)
    if j is not None and isinstance(j, (list, tuple)):
        out = []
        for item in j:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    start = int(item[0])
                    dur = int(item[1])
                    tgt = item[2]
                    # Handle nested vector
                    if isinstance(tgt, (list, tuple)):
                        if len(tgt) >= 3:
                            tx, ty, tz = float(tgt[0]), float(tgt[1]), float(tgt[2])
                        else:
                            continue  # Skip invalid
                    else:
                        # Parse as string format
                        tx, ty, tz = parse_vector3(str(tgt))
                    out.append((start, dur, (tx, ty, tz)))
            except Exception:
                continue
        return sorted(out, key=lambda x: x[0])
    # Fallback: semicolon-separated format
    s = text.strip()
    if s == "":
        return []
    # Handle outer curly braces if present
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1].strip()
    segments = []
    cur = ""
    depth = 0
    # Parse segments with proper bracket handling
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        cur += ch
        if depth == 0 and (ch in ');' or (ch == ',' and cur.strip())):
            segments.append(cur.strip())
            cur = ""
    if cur.strip() != "":
        segments.append(cur.strip())
    out = []
    for seg in segments:
        # Clean segment
        seg_clean = seg.strip()
        for ch in '()[]{}':
            seg_clean = seg_clean.replace(ch, '')
        seg_clean = seg_clean.strip()
        if not seg_clean:
            continue
        parts = [p.strip() for p in seg_clean.split(';') if p.strip() != '']
        # Expect at least 5 parts (start;duration;x;y;z)
        if len(parts) >= 5:
            try:
                start = int(parts[0])
                dur = int(parts[1])
                tx = float(parts[2])
                ty = float(parts[3])
                tz = float(parts[4])
                out.append((start, dur, (tx, ty, tz)))
            except Exception:
                continue
    return sorted(out, key=lambda x: x[0])

# ----------------------
# Line change parsing
# ----------------------
def parse_line_changes_field(text: str) -> List[Tuple[int,int,Tuple[float,float,float], float]]:
    """Parse line changes format: (start;duration;x;y;z;power)"""
    if text is None:
        return []
    # Try JSON parsing
    j = try_parse_json(text)
    if j is not None and isinstance(j, (list, tuple)):
        out = []
        for item in j:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    start = int(item[0])
                    dur = int(item[1])
                    third = item[2]
                    # Handle nested vector with power
                    if isinstance(third, (list, tuple)) and len(third) >= 4:
                        x, y, z, p = [float(third[i]) for i in range(4)]
                    elif len(item) >= 6:  # Flattened format
                        x, y, z = float(item[2]), float(item[3]), float(item[4])
                        p = float(item[5])
                    else:
                        continue
                    out.append((start, dur, (x, y, z), float(p)))
            except Exception:
                continue
        return sorted(out, key=lambda x: x[0])
    # Fallback: semicolon-separated format
    s = text.strip()
    if s == "":
        return []
    # Handle outer curly braces
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1].strip()
    segments = []
    cur = ""
    depth = 0
    # Parse segments with bracket awareness
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        cur += ch
        if depth == 0 and (ch in ');' or (ch == ',' and cur.strip())):
            segments.append(cur.strip())
            cur = ""
    if cur.strip() != "":
        segments.append(cur.strip())
    out = []
    for seg in segments:
        seg_clean = seg.strip()
        for ch in '()[]{}':
            seg_clean = seg_clean.replace(ch, '')
        seg_clean = seg_clean.strip()
        if not seg_clean:
            continue
        parts = [p.strip() for p in seg_clean.split(';') if p.strip() != '']
        # Expect 6 parts: start;duration;x;y;z;power
        if len(parts) >= 6:
            try:
                start = int(parts[0])
                dur = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                p = float(parts[5])
                out.append((start, dur, (x, y, z), p))
            except Exception:
                continue
    return sorted(out, key=lambda x: x[0])

# ----------------------
# Color change parsing
# ----------------------
def parse_color_changes(text: str) -> List[Tuple[int,int,Tuple[float,float,float]]]:
    """Parse color changes format: (start;duration;r;g;b)"""
    if text is None:
        return []
    # Try JSON parsing
    j = try_parse_json(text)
    if j is not None and isinstance(j, (list, tuple)):
        out = []
        for item in j:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    start = int(item[0])
                    dur = int(item[1])
                    third = item[2]
                    # Handle color vector
                    if isinstance(third, (list, tuple)) and len(third) >= 3:
                        r, g, b = [float(third[i]) for i in range(3)]
                    else:
                        colors = parse_vectorN(str(third))
                        if len(colors) >= 3:
                            r, g, b = colors[0], colors[1], colors[2]
                        else:
                            continue
                    out.append((start, dur, (r, g, b)))
            except Exception:
                continue
        return sorted(out, key=lambda x: x[0])
    # Fallback: semicolon-separated format
    s = text.strip()
    if s == "":
        return []
    # Handle outer curly braces
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1].strip()
    segments = []
    cur = ""
    depth = 0
    # Parse segments with bracket awareness
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        cur += ch
        if depth == 0 and (ch in ');' or (ch == ',' and cur.strip())):
            segments.append(cur.strip())
            cur = ""
    if cur.strip() != "":
        segments.append(cur.strip())
    out = []
    for seg in segments:
        seg_clean = seg.strip()
        for ch in '()[]{}':
            seg_clean = seg_clean.replace(ch, '')
        seg_clean = seg_clean.strip()
        if not seg_clean:
            continue
        parts = [p.strip() for p in seg_clean.split(';') if p.strip() != '']
        # Expect 5 parts: start;duration;r;g;b
        if len(parts) >= 5:
            try:
                start = int(parts[0])
                dur = int(parts[1])
                r = float(parts[2])
                g = float(parts[3])
                b = float(parts[4])
                out.append((start, dur, (r, g, b)))
            except Exception:
                continue
    return sorted(out, key=lambda x: x[0])

# ----------------------
# Bézier sampling
# ----------------------
def sample_quadratic_bezier(p0: Tuple[float,float,float], 
                           pc: Tuple[float,float,float], 
                           p1: Tuple[float,float,float], 
                           n_samples: int = 16) -> List[Tuple[float,float,float]]:
    """Sample points along quadratic Bézier curve"""
    samples = []
    if n_samples < 2:
        n_samples = 2
    for i in range(n_samples + 1):
        t = i / n_samples
        x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * pc[0] + t**2 * p1[0]
        y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * pc[1] + t**2 * p1[1]
        z = (1 - t)**2 * p0[2] + 2 * (1 - t) * t * pc[2] + t**2 * p1[2]
        samples.append((x, y, z))
    return samples

def sample_count_for_curve(p0: Tuple[float,float,float], 
                           pc: Tuple[float,float,float], 
                           p1: Tuple[float,float,float], 
                           min_samples: int = 8) -> int:
    """Determine appropriate sample count based on curve length"""
    approx_len = vec_distance(p0, pc) + vec_distance(pc, p1)
    # Calculate samples (0.5 samples per unit length)
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
        self.points = {}   # uuid -> { base: (x,y,z), connections: [...], movements: [...] }
        self.lines = {}    # uuid -> { endpoints: [u1,u2], pull_base: (x,y,z), power_base: float, changes: [...] }
        self.shapes = {}   # uuid -> { points: [...], lines: [...], fill_base: (r,g,b), color_changes: [...] }
        self._load_all()

    def _load_all(self):
        """Load all data from database into memory structures"""
        cur = self.conn.cursor()
        # Load points
        try:
            rows = cur.execute("SELECT uuid, coordinates, connected_points, movements FROM points").fetchall()
            for r in rows:
                try:
                    uuid = str(r["uuid"])
                    base = parse_vector3(r["coordinates"])
                    conns = parse_uuid_list(r["connected_points"]) if r["connected_points"] is not None else []
                    moves = parse_movements_field(r["movements"]) if r["movements"] is not None else []
                    self.points[uuid] = {
                        "base": base, 
                        "connections": conns, 
                        "movements": moves
                    }
                except Exception as e:
                    print(f"Warning: skipping malformed point {r.get('uuid', 'unknown')}: {e}")
        except sqlite3.OperationalError:
            print("Warning: points table not found or invalid structure")
        
        # Load lines
        try:
            rows = cur.execute("SELECT uuid, endpoints, pull_point, pull_power, movements FROM lines").fetchall()
            for r in rows:
                try:
                    uuid = str(r["uuid"])
                    endpoints = parse_uuid_list(r["endpoints"])
                    pull_base = parse_vector3(r["pull_point"])
                    power_base = float(r["pull_power"]) if r["pull_power"] is not None else 1.0
                    changes = parse_line_changes_field(r["movements"]) if r["movements"] is not None else []
                    self.lines[uuid] = {
                        "endpoints": endpoints, 
                        "pull_base": pull_base, 
                        "power_base": power_base, 
                        "changes": changes
                    }
                except Exception as e:
                    print(f"Warning: skipping malformed line {r.get('uuid', 'unknown')}: {e}")
        except sqlite3.OperationalError:
            print("Warning: lines table not found or invalid structure")
        
        # Load shapes (FIXED: Now correctly uses uuid column)
        try:
            rows = cur.execute("SELECT uuid, point_uuids, line_uuids, color, movements FROM shapes").fetchall()
            for r in rows:
                try:
                    uuid = str(r["uuid"])  # FIXED: Now properly getting uuid
                    pts = parse_uuid_list(r["point_uuids"])
                    lns = parse_uuid_list(r["line_uuids"])
                    fill = tuple(parse_vectorN(r["color"])[:3]) if r["color"] is not None else (1.0, 1.0, 1.0)
                    color_changes = parse_color_changes(r["movements"]) if r["movements"] is not None else []
                    self.shapes[uuid] = {
                        "points": pts, 
                        "lines": lns, 
                        "fill_base": fill, 
                        "color_changes": color_changes
                    }
                except Exception as e:
                    print(f"Warning: skipping malformed shape {r.get('uuid', 'unknown')}: {e}")
        except sqlite3.OperationalError:
            print("Warning: shapes table not found or invalid structure")

    # ----------------------
    # compute point position at tick
    # ----------------------
    def point_position_at(self, uuid: str, tick_ms: int) -> Tuple[float,float,float]:
        """Calculate point position at given time"""
        rec = self.points.get(uuid)
        if rec is None:
            raise KeyError(f"Point {uuid} not found")
        base = rec["base"]
        movements = rec["movements"]
        current = base
        for m in movements:
            start, dur, target = m
            end = start + dur
            # Before movement starts
            if tick_ms < start:
                return current
            # Instant movement
            if dur <= 0:
                current = target
                continue
            # After movement completes
            if tick_ms >= end:
                current = target
                continue
            # During movement
            progress = (tick_ms - start) / dur
            return lerp_vec(current, target, clamp(progress))
        return current

    # ----------------------
    # compute pull point and power at tick for a line
    # ----------------------
    def line_pull_at(self, line_uuid: str, tick_ms: int) -> Tuple[Tuple[float,float,float], float]:
        """Calculate line pull point and power at given time"""
        rec = self.lines.get(line_uuid)
        if rec is None:
            raise KeyError(f"Line {line_uuid} not found")
        base = rec["pull_base"]
        base_power = rec["power_base"]
        changes = rec["changes"]
        current_pos = base
        current_power = base_power
        for ch in changes:
            start, dur, new_coords, new_power = ch
            end = start + dur
            if tick_ms < start:
                return current_pos, current_power
            if dur <= 0:
                current_pos = new_coords
                current_power = new_power
                continue
            if tick_ms >= end:
                current_pos = new_coords
                current_power = new_power
                continue
            progress = (tick_ms - start) / dur
            curp = lerp_vec(current_pos, new_coords, clamp(progress))
            curpow = lerp(current_power, new_power, clamp(progress))
            return curp, curpow
        return current_pos, current_power

    # ----------------------
    # compute shape color at tick
    # ----------------------
    def shape_color_at(self, shape_uuid: str, tick_ms: int) -> Tuple[float,float,float]:
        """Calculate shape fill color at given time"""
        rec = self.shapes.get(shape_uuid)
        if rec is None:
            raise KeyError(f"Shape {shape_uuid} not found")
        base = rec["fill_base"]
        changes = rec["color_changes"]
        current = base
        for ch in changes:
            start, dur, newcol = ch
            end = start + dur
            if tick_ms < start:
                return current
            if dur <= 0:
                current = newcol
                continue
            if tick_ms >= end:
                current = newcol
                continue
            progress = (tick_ms - start) / dur
            return (
                lerp(current[0], newcol[0], clamp(progress)),
                lerp(current[1], newcol[1], clamp(progress)),
                lerp(current[2], newcol[2], clamp(progress))
            )
        return current

    # ----------------------
    # Public: build full frame
    # ----------------------
    def read_full(self, tick_in_ms: int) -> Dict[str, Any]:
        """Generate complete frame data for rendering at given time"""
        # 1) Compute all point positions
        point_positions = {}
        for uuid in self.points:
            try:
                point_positions[uuid] = self.point_position_at(uuid, tick_in_ms)
            except KeyError:
                continue  # Skip missing points
        
        # 2) Compute lines
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
            
            # FIXED: Implement pull_power to scale control point influence
            # Calculate midpoint
            mid = ((p0[0] + p1[0])/2, (p0[1] + p1[1])/2, (p0[2] + p1[2])/2)
            # Calculate offset from midpoint to pull_point
            offset = (pull_point[0] - mid[0], pull_point[1] - mid[1], pull_point[2] - mid[2])
            # Scale offset by pull_power
            scaled_offset = (offset[0] * pull_power, offset[1] * pull_power, offset[2] * pull_power)
            # Calculate effective control point
            effective_pc = (mid[0] + scaled_offset[0], mid[1] + scaled_offset[1], mid[2] + scaled_offset[2])
            
            # Sample the Bézier curve using effective control point
            n_samples = sample_count_for_curve(p0, effective_pc, p1)
            samples = sample_quadratic_bezier(p0, effective_pc, p1, n_samples)
            
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
        
        # 3) Compute shapes
        shapes_out = []
        for uuid, rec in self.shapes.items():
            # Get valid point positions
            coords_for_fan = []
            for pu in rec["points"]:
                if pu in point_positions:
                    coords_for_fan.append(point_positions[pu])
            # Triangulate using fan method
            triangles = []
            if len(coords_for_fan) >= 3:
                for i in range(1, len(coords_for_fan) - 1):
                    triangles.append([
                        coords_for_fan[0],
                        coords_for_fan[i],
                        coords_for_fan[i + 1]
                    ])
            color = self.shape_color_at(uuid, tick_in_ms)
            shapes_out.append({
                "uuid": uuid,
                "triangles": triangles,
                "color": color
            })
        
        # 4) Prepare points list
        points_out = [{"uuid": u, "pos": pos} for u, pos in point_positions.items()]
        return {
            "timestamp_ms": tick_in_ms,
            "points": points_out,
            "lines": lines_out,
            "shapes": shapes_out
        }

# ----------------------
# Example quick test
# ----------------------
def _example_in_memory_db():
    """Create an in-memory database with sample data"""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    # Create tables with the correct column names as defined in ZeroInit.py
    cur.execute("CREATE TABLE points(uuid TEXT PRIMARY KEY, coordinates TEXT NOT NULL, connected_points TEXT, movements TEXT)")
    cur.execute("CREATE TABLE lines(uuid TEXT PRIMARY KEY, endpoints TEXT NOT NULL, pull_point TEXT NOT NULL, pull_power REAL NOT NULL, movements TEXT)")
    cur.execute("CREATE TABLE shapes(uuid TEXT PRIMARY KEY, point_uuids TEXT NOT NULL, line_uuids TEXT NOT NULL, color TEXT NOT NULL, movements TEXT)")
    # Insert sample points
    cur.execute("INSERT INTO points VALUES(?, ?, ?, ?)",
                ("A", json.dumps([0.0,0.0,0.0]), json.dumps(["B"]), json.dumps([[0, 1000, [0.0, 1.0, 0.0]]])))
    cur.execute("INSERT INTO points VALUES(?, ?, ?, ?)",
                ("B", json.dumps([2.0,0.0,0.0]), json.dumps(["A"]), json.dumps([[500, 1000, [2.0, 1.0, 0.0]]])))
    cur.execute("INSERT INTO points VALUES(?, ?, ?, ?)",
                ("C", json.dumps([1.0, 0.5, 0.0]), json.dumps([]), None))
    # Insert sample line
    cur.execute("INSERT INTO lines VALUES(?, ?, ?, ?, ?)",
                ("L1", json.dumps(["A","B"]), json.dumps([1.0, 1.0, 1.5]), 1.0, 
                 json.dumps([[0, 1000, [1.0, 1.0, 1.5]]])))
    # Insert sample shape (FIXED: Added uuid)
    cur.execute("INSERT INTO shapes VALUES(?, ?, ?, ?, ?)",
                ("S1", json.dumps(["A","B","C"]), json.dumps(["L1"]), json.dumps([0.2,0.7,0.3]), None))
    conn.commit()
    return conn

def run_example():
    """Run a demonstration of the interpreter"""
    conn = _example_in_memory_db()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn.backup(sqlite3.connect(tf.name))
        interp = ZeroInterpreter(tf.name)
        for t in [0, 250, 500, 750, 1000, 1500]:
            frame = interp.read_full(t)
            print(f"\nTime: {t}ms")
            print(f"  Points: {len(frame['points'])}")
            print(f"  Lines: {len(frame['lines'])}")
            print(f"  Shapes: {len(frame['shapes'])}")
            if frame['lines']:
                samples = frame['lines'][0]['samples']
                print(f"  Line samples: {len(samples)} (first: {samples[0]}, last: {samples[-1]})")

if __name__ == "__main__":
    run_example()