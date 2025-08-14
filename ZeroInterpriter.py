# ZeroInterpreter.py - Connects to an SQLite DB with the tables:
# points(uuid TEXT, coordinates TEXT, connected_points TEXT, movements TEXT)
# lines(uuid TEXT, endpoints TEXT, pull_point TEXT, pull_power REAL, movements TEXT)
# shapes(uuid TEXT, point_uuids TEXT, line_uuids TEXT, color TEXT, movements TEXT) # FIXED: Added uuid
# music(timestamp_ms INTEGER, notes TEXT, durations TEXT, instrument_id TEXT)
# speech(id INTEGER, sentence TEXT, start_time_ms INTEGER, voice_id TEXT)
# - Supports two field formats:
#   * JSON-like arrays: [x,y,z] or [ [...],... ]
#   * semicolon-separated: "x;y;z" or "(x;y;[a;b;c]);..." etc.

import sqlite3
import json
import uuid
import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Utility functions
def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t)

def lerp_vec(a: Tuple[float,float,float], b: Tuple[float,float,float], t: float) -> Tuple[float,float,float]:
    return (
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t)
    )

def parse_vector3_from_any(s: Any) -> Tuple[float, float, float]:
    """Accepts: JSON list/tuple or string like "[x,y,z]" or "x;y;z" or "(x;y;z)" or "x,y,z"
    Returns tuple (x,y,z) or raises ValueError."""
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
    
    # split by comma or semicolon
    parts = [p.strip() for p in re.split(r'[,;]', txt) if p.strip() != '']
    if len(parts) >= 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    
    raise ValueError(f"Cannot parse '{s}' as vector3")

def try_load_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except:
        return None

def normalize_uuid_list_json( Any) -> str:
    """Ensure data is in proper JSON format for uuid lists"""
    if isinstance(data, str):
        # Try to see if it's already JSON
        try:
            json.loads(data)
            return data  # Already valid JSON
        except:
            # Maybe it's a semicolon-separated list
            uuids = [u.strip() for u in data.split(';') if u.strip()]
            return json.dumps(uuids)
    elif isinstance(data, (list, tuple)):
        return json.dumps([str(u) for u in data])
    else:
        return json.dumps([])

def normalize_vector3_json(data: Any) -> str:
    """Ensure data is in proper JSON format for vector3"""
    try:
        x, y, z = parse_vector3_from_any(data)
        return json.dumps([x, y, z])
    except:
        return json.dumps([0.0, 0.0, 0.0])

def normalize_line_changes_json( Any) -> str:
    """Ensure data is in proper JSON format for line changes"""
    if data is None:
        return json.dumps([])
    
    if isinstance(data, str):
        # Try to see if it's already JSON
        try:
            json.loads(data)
            return data  # Already valid JSON
        except:
            # Parse the semicolon format
            segments = []
            s = data.strip()
            if s.startswith('{') and s.endswith('}'):
                s = s[1:-1].strip()
            
            cur = ""
            depth = 0
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
                parts = [p.strip() for p in seg.split(';') if p.strip() != '']
                if len(parts) >= 4:
                    try:
                        start = int(float(parts[0]))
                        dur = int(float(parts[1]))
                        x, y, z = parse_vector3_from_any(parts[2])
                        p = float(parts[3])
                        out.append([start, dur, [x, y, z, p]])
                    except:
                        continue
            return json.dumps(out)
    
    elif isinstance(data, (list, tuple)):
        out = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                start = int(item[0])
                dur = int(item[1])
                third = item[2]
                if isinstance(third, (list, tuple)) and len(third) >= 4:
                    x, y, z, p = float(third[0]), float(third[1]), float(third[2]), float(third[3])
                elif len(item) >= 6:
                    x, y, z, p = float(item[2]), float(item[3]), float(item[4]), float(item[5])
                else:
                    continue
                out.append([start, dur, [x, y, z, p]])
        return json.dumps(out)
    
    return json.dumps([])

def parse_movements_field(text: str) -> List[Tuple[int,int,Tuple[float,float,float]]]:
    """Parse movements field which should be in format:
    [ [start_ms, duration_ms, [x,y,z]], ... ]"""
    if text is None:
        return []
    
    if isinstance(text, (list, tuple)):
        out = []
        for item in text:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                start = int(item[0])
                dur = int(item[1])
                third = item[2]
                if isinstance(third, (list, tuple)) and len(third) >= 3:
                    x, y, z = float(third[0]), float(third[1]), float(third[2])
                elif len(item) >= 5:
                    x, y, z = float(item[2]), float(item[3]), float(item[4])
                else:
                    continue
                out.append((start, dur, (x, y, z)))
        return out
    
    txt = str(text).strip()
    if txt == "":
        return []
    
    # find bracketed segments
    segments = re.findall(r'[\(\[\{]([^}\]\)]*)[\)\]\}]', txt)
    out = []
    for seg in segments:
        parts = [p.strip() for p in seg.split(';') if p.strip() != '']
        if len(parts) >= 5:
            try:
                start = int(float(parts[0]))
                dur = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                out.append((start, dur, (x, y, z)))
            except Exception:
                continue
    
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
        seg_clean = seg
        for ch in '(){}[]': 
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

def parse_line_changes_field(text: str) -> List[Tuple[int,int,Tuple[float,float,float], float]]:
    """Parse line changes field which should be in format:
    [ [start_ms, duration_ms, [x,y,z,power]], ... ]"""
    if text is None:
        return []
    
    if isinstance(text, (list, tuple)):
        out = []
        for item in text:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                start = int(item[0])
                dur = int(item[1])
                third = item[2]
                if isinstance(third, (list, tuple)) and len(third) >= 4:
                    x, y, z, p = float(third[0]), float(third[1]), float(third[2]), float(third[3])
                elif len(item) >= 6:
                    x, y, z, p = float(item[2]), float(item[3]), float(item[4]), float(item[5])
                else:
                    continue
                out.append((start, dur, (x, y, z), float(p)))
        return out
    
    txt = str(text).strip()
    if txt == "":
        return []
    
    segments = re.findall(r'[\(\[\{]([^}\]\)]*)[\)\]\}]', txt)
    out = []
    for seg in segments:
        parts = [p.strip() for p in seg.split(';') if p.strip() != '']
        if len(parts) >= 5:
            try:
                start = int(float(parts[0]))
                dur = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                p = float(parts[5]) if len(parts) > 5 else 1.0
                out.append((start, dur, (x, y, z), float(p)))
            except:
                continue
    
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
        seg_clean = seg
        for ch in '(){}[]': 
            seg_clean = seg_clean.replace(ch, '')
        seg_clean = seg_clean.strip()
        if not seg_clean:
            continue
        parts = [p.strip() for p in seg_clean.split(';') if p.strip() != '']
        # Expect at least 6 parts (start;duration;x;y;z;power)
        if len(parts) >= 6:
            try:
                start = int(parts[0])
                dur = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                p = float(parts[5])
                out.append((start, dur, (x, y, z), float(p)))
            except Exception:
                continue
    return sorted(out, key=lambda x: x[0])

def parse_color_changes(text: str) -> List[Tuple[int,int,Tuple[float,float,float]]]:
    """Parse color changes field which should be in format:
    [ [start_ms, duration_ms, [r,g,b]], ... ]"""
    if text is None:
        return []
    
    if isinstance(text, (list, tuple)):
        out = []
        for item in text:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                start = int(item[0])
                dur = int(item[1])
                col = item[2]
                r, g, b = parse_vector3_from_any(col)
                out.append([start, dur, [r, g, b]])
        return out
    
    txt = str(text).strip()
    if txt == "":
        return []
    
    segments = re.findall(r'[\(\[\{]([^}\]\)]*)[\)\]\}]', txt)
    out = []
    for seg in segments:
        parts = [p.strip() for p in seg.split(';') if p.strip() != '']
        if len(parts) >= 5:
            try:
                start = int(float(parts[0]))
                dur = int(float(parts[1]))
                r = float(parts[2])
                g = float(parts[3])
                b = float(parts[4])
                out.append([start, dur, [r, g, b]])
            except:
                continue
    
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
        seg_clean = seg
        for ch in '(){}[]': 
            seg_clean = seg_clean.replace(ch, '')
        seg_clean = seg_clean.strip()
        if not seg_clean:
            continue
        parts = [p.strip() for p in seg_clean.split(';') if p.strip() != '']
        # Expect at least 5 parts (start;duration;r;g;b)
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

def extract_table_block(text: str, table_marker: str) -> str:
    """Return the block of text after the first occurrence of table_marker
    (e.g. "TABLE1") up to the next TABLE[1-3] marker or EOF.
    Examples:
    extract_table_block(raw_text, "TABLE1")
    Will return the text immediately after "TABLE1" (case-insensitive)
    until the next "TABLE1/2/3" header or end of file."""
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
        return text[start:start+nxt.start()].strip()
    else:
        return text[start:].strip()

def parse_legacy_rows(text: str) -> List[List[str]]:
    """Parse legacy text format into rows of fields.
    Legacy format: one row per line, fields separated by '|'
    Empty lines and comment lines (starting with '#') are ignored."""
    rows = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        fields = [f.strip() for f in line.split('|')]
        rows.append(fields)
    return rows

class ZeroInterpreter:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.points = {}
        self.lines = {}
        self.shapes = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all data from the database into memory structures"""
        cur = self.conn.cursor()
        
        # Load points
        try:
            rows = cur.execute("SELECT uuid, coordinates, connected_points, movements FROM points").fetchall()
            for r in rows:
                try:
                    uuid = str(r["uuid"])
                    base = parse_vector3_from_any(r["coordinates"])
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
                    # FIXED: Ensure color is always a 3-tuple
                    try:
                        fill = parse_vector3(r["color"])
                    except (ValueError, TypeError):
                        fill = (1.0, 1.0, 1.0)
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
    
    def get_music_events(self, start_time_ms: int, end_time_ms: int = None) -> List[Dict[str, Any]]:
        """Get music events that occur within the specified time range
        Args:
            start_time_ms: Start of time range in milliseconds
            end_time_ms: End of time range in milliseconds (defaults to start_time_ms)
        Returns:
            List of music events in the time range"""
        if end_time_ms is None:
            end_time_ms = start_time_ms
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT timestamp_ms, notes, durations, instrument_id 
            FROM music 
            WHERE timestamp_ms <= ? AND (timestamp_ms + CAST(json_extract(durations, '$[0]') AS INTEGER)) > ?
        """, (end_time_ms, start_time_ms))
        
        events = []
        for row in cur.fetchall():
            try:
                notes = json.loads(row["notes"])
                durations = json.loads(row["durations"])
                events.append({
                    "timestamp_ms": row["timestamp_ms"],
                    "notes": notes,
                    "durations": durations,
                    "instrument_id": row["instrument_id"]
                })
            except Exception as e:
                print(f"Warning: skipping malformed music event at {row['timestamp_ms']}: {e}")
        
        return events
    
    def get_speech_events(self, start_time_ms: int, end_time_ms: int = None) -> List[Dict[str, Any]]:
        """Get speech events that occur within the specified time range
        Args:
            start_time_ms: Start of time range in milliseconds
            end_time_ms: End of time range in milliseconds (defaults to start_time_ms)
        Returns:
            List of speech events in the time range"""
        if end_time_ms is None:
            end_time_ms = start_time_ms
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, sentence, start_time_ms, voice_id 
            FROM speech 
            WHERE start_time_ms >= ? AND start_time_ms <= ?
        """, (start_time_ms, end_time_ms))
        
        events = []
        for row in cur.fetchall():
            events.append({
                "id": row["id"],
                "sentence": row["sentence"],
                "start_time_ms": row["start_time_ms"],
                "voice": row["voice_id"]
            })
        return events
    
    # -# compute point position at tick# -
    def point_position_at(self, uuid: str, tick_ms: int) -> Tuple[float,float,float]:
        """Calculate point position at given time
        NOTE: Movements for a single point must not overlap (enforced by documentation).
        If they do overlap, behavior is undefined and the input should be considered invalid."""
        rec = self.points.get(uuid)
        if rec is None:
            raise KeyError(f"Point {uuid} not found")
        
        base = rec["base"]
        movements = rec["movements"]
        current = base
        
        for m in movements:
            start, dur, target = m
            end = start + dur
            
            # Skip movements that haven't started yet
            if tick_ms < start:
                continue
                
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
            current = lerp_vec(current, target, clamp(progress))
        
        return current
    
    # -# compute pull point and power at tick for a line# -
    def line_pull_at(self, line_uuid: str, tick_ms: int) -> Tuple[Tuple[float,float,float], float]:
        """Calculate line pull point and power at given time
        NOTE: Line changes must not overlap (enforced by documentation).
        If they do overlap, behavior is undefined and the input should be considered invalid."""
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
                continue
                
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
    
    # -# compute shape color at tick# -
    def shape_color_at(self, shape_uuid: str, tick_ms: int) -> Tuple[float,float,float]:
        """Calculate shape fill color at given time
        NOTE: Color changes must not overlap (enforced by documentation).
        If they do overlap, behavior is undefined and the input should be considered invalid."""
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
                continue
                
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
    
    # -# Public: build full frame# -
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
            # Skip lines with insufficient endpoints
            if len(endpoints) < 2:
                continue
            
            u0, u1 = endpoints[0], endpoints[1]
            if u0 not in point_positions or u1 not in point_positions:
                continue
            
            p0 = point_positions[u0]
            p1 = point_positions[u1]
            
            # Calculate pull point and power
            pull_point, pull_power = self.line_pull_at(uuid, tick_in_ms)
            
            lines_out.append({
                "uuid": uuid,
                "points": [p0, p1],
                "pull_point": pull_point,
                "pull_power": pull_power
            })
        
        # 3) Compute shapes
        shapes_out = []
        for uuid, rec in self.shapes.items():
            point_uuids = rec["points"]
            line_uuids = rec["lines"]
            
            # Get positions for all points in the shape
            coords = []
            for p_uuid in point_uuids:
                if p_uuid in point_positions:
                    coords.append(point_positions[p_uuid])
            
            # Triangulate the polygon (simple fan triangulation for convex polygons)
            triangles = []
            if len(coords) >= 3:
                for i in range(1, len(coords) - 1):
                    triangles.append([coords[0], coords[i], coords[i + 1]])
            
            shapes_out.append({
                "uuid": uuid,
                "points": point_uuids,
                "lines": line_uuids,
                "color": self.shape_color_at(uuid, tick_in_ms),
                "triangles": triangles
            })
        
        return {
            "points": point_positions,
            "lines": lines_out,
            "shapes": shapes_out,
            "time_ms": tick_in_ms
        }