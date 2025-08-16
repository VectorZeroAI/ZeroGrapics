# ZeroInterpreter.py - Connects to an SQLite DB with the tables:
# points(coordinates TEXT, uuid TEXT PRIMARY KEY, connected_points TEXT, movements TEXT)
# lines(uuid TEXT PRIMARY KEY, endpoints TEXT, pull_point TEXT, pull_power REAL, movements TEXT)
# shapes(uuid TEXT PRIMARY KEY, point_uuids TEXT, line_uuids TEXT, color TEXT, movements TEXT)
# music(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_ms INTEGER NOT NULL, notes TEXT NOT NULL, durations TEXT NOT NULL, instrument_id TEXT NOT NULL)
# speech(id INTEGER PRIMARY KEY AUTOINCREMENT, sentence TEXT NOT NULL, start_time_ms INTEGER NOT NULL, voice_id TEXT NOT NULL)
# Exposes the funktions used to comunicate with the DB system.
# Exposes funktions:
# "construct_the_model"
# "read.points"
# "read.lines"
# "read.shapes"
# "read.movements.lines"
# "read.movements.points"
# "read.movements.shapes"
# "read.full(tick_in_ms)"
# read.full(tick_in_ms) returns the state that should be rendered in that exact ms.
import sqlite3
import json
import math
from typing import Any, Dict, List, Tuple, Union, Optional

# --- Utility Functions ---

def clamp(x: float) -> float:
    """Clamps a value between 0.0 and 1.0."""
    return max(0.0, min(1.0, x))

def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolates between a and b by factor t."""
    return a + (b - a) * clamp(t)

def lerp_vec(a: Tuple[float, float, float], b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    """Linearly interpolates between two 3D vectors."""
    return (
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t)
    )

def parse_vector3(s: Any) -> Tuple[float, float, float]:
    """Parses a string like '{1.0;2.0;3.0}' or '[1.0,2.0,3.0]' into a tuple (1.0, 2.0, 3.0).
    Handles strings, tuples, lists, and returns (0,0,0) on error.
    Prioritizes JSON parsing.
    """
    if isinstance(s, (tuple, list)) and len(s) >= 3:
        try:
            return tuple(float(x) for x in s[:3]) # Take first 3 elements
        except (ValueError, TypeError):
            pass
    if isinstance(s, str):
        s = s.strip()
        # Try JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                 return tuple(float(x) for x in parsed[:3])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Fallback to semicolon parsing
        try:
            parts = s.strip('{}()[]').split(';')
            if len(parts) >= 3:
                return tuple(float(p) for p in parts[:3]) # Take first 3 parts
        except (ValueError, AttributeError):
            pass
    return (0.0, 0.0, 0.0)

def parse_uuid_list(s: Any) -> List[str]:
    """Parses a string like '{uuid1;uuid2}' or '["uuid1", "uuid2"]' into a list ['uuid1', 'uuid2'].
    Handles strings, lists, and returns [] on error.
    Prioritizes JSON parsing.
    """
    if isinstance(s, list):
        return [str(item) for item in s]
    if isinstance(s, tuple): # Handle tuples as lists
        return [str(item) for item in s]
    if isinstance(s, str):
        s = s.strip()
        # Try JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback to semicolon parsing
        try:
            stripped = s.strip('{}()[]')
            if not stripped:
                return []
            # Split by semicolon or comma
            delimiter = ';' if ';' in stripped else ','
            return [uuid.strip() for uuid in stripped.split(delimiter) if uuid.strip()]
        except AttributeError:
            pass
    return []

def try_parse_json(s: str) -> Optional[Any]:
    """Attempts to parse a string as JSON. Returns the parsed object or None."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None

def parse_movements_field(text: str) -> List[Tuple[int, int, Tuple[float, float, float]]]:
    """
    Parse movements field which should be in format:
    JSON: [ [start_ms, duration_ms, [x,y,z]], ... ]
    Fallback: [ [start_ms, duration_ms, {x;y;z}], ... ] or "{(start;duration;[x;y;z]);...}"
    """
    if text is None or text == "":
        return []
    # Try JSON parsing first (this is what ZeroFiller stores)
    data = try_parse_json(text)
    if isinstance(data, (list, tuple)):
        out = []
        for item in data:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    start = int(item[0])
                    dur = int(item[1])
                    # Third element is the target vector
                    target_data = item[2]
                    if isinstance(target_data, (list, tuple)) and len(target_data) >= 3:
                        x, y, z = float(target_data[0]), float(target_data[1]), float(target_data[2])
                    else:
                        # Fallback: parse if it's a string representation
                        x, y, z = parse_vector3(target_data)
                    out.append((start, dur, (x, y, z)))
            except (ValueError, TypeError, IndexError):
                continue # Skip malformed items
        return sorted(out, key=lambda x: x[0])

    # Fallback: Legacy semicolon-separated format parsing
    txt = str(text).strip()
    if txt.startswith('{') and txt.endswith('}'):
        txt = txt[1:-1].strip()
    segments = []
    cur = ""
    depth = 0
    for ch in txt:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        cur += ch
        if depth == 0 and ch in ');':
            segments.append(cur.strip('();'))
            cur = ""
    if cur.strip('();'):
        segments.append(cur.strip('();'))

    out = []
    for seg in segments:
        try:
            parts = [p.strip() for p in seg.split(';') if p.strip()]
            if len(parts) >= 5: # start;dur;x;y;z
                start = int(parts[0])
                dur = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                out.append((start, dur, (x, y, z)))
        except (ValueError, IndexError):
            continue
    return sorted(out, key=lambda x: x[0])

def parse_line_changes_field(text: str) -> List[Tuple[int, int, Tuple[float, float, float], float]]:
    """
    Parse line changes field which should be in format:
    JSON: [ [start_ms, duration_ms, [x,y,z,power]], ... ]
    Fallback: "{(start;duration;[x;y;z;power]);...}"
    """
    if text is None or text == "":
        return []
    # Try JSON parsing first (this is what ZeroFiller stores)
    data = try_parse_json(text)
    if isinstance(data, (list, tuple)):
        out = []
        for item in data:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    start = int(item[0])
                    dur = int(item[1])
                    # Third element is the target vector + power
                    target_data = item[2]
                    if isinstance(target_data, (list, tuple)) and len(target_data) >= 4:
                        x, y, z, p = float(target_data[0]), float(target_data[1]), float(target_data[2]), float(target_data[3])
                    # Or if item has 6 elements directly: [start, dur, x, y, z, p]
                    elif len(item) >= 6:
                         x, y, z, p = float(item[2]), float(item[3]), float(item[4]), float(item[5])
                    else:
                        continue
                    out.append((start, dur, (x, y, z), p))
            except (ValueError, TypeError, IndexError):
                continue # Skip malformed items
        return sorted(out, key=lambda x: x[0])

    # Fallback: Legacy semicolon-separated format parsing
    txt = str(text).strip()
    if txt.startswith('{') and txt.endswith('}'):
        txt = txt[1:-1].strip()
    segments = []
    cur = ""
    depth = 0
    for ch in txt:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        cur += ch
        if depth == 0 and ch in ');':
            segments.append(cur.strip('();'))
            cur = ""
    if cur.strip('();'):
        segments.append(cur.strip('();'))

    out = []
    for seg in segments:
        try:
            parts = [p.strip() for p in seg.split(';') if p.strip()]
            if len(parts) >= 6: # start;dur;x;y;z;power
                start = int(parts[0])
                dur = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                p = float(parts[5])
                out.append((start, dur, (x, y, z), p))
        except (ValueError, IndexError):
            continue
    return sorted(out, key=lambda x: x[0])

def parse_color_changes(text: str) -> List[Tuple[int, int, Tuple[float, float, float]]]:
    """
    Parse color changes field which should be in format:
    JSON: [ [start_ms, duration_ms, [r,g,b]], ... ]
    Fallback: "{(start;duration;[r;g;b]);...}"
    """
    if text is None or text == "":
        return []
    # Try JSON parsing first (this is what ZeroFiller stores)
    data = try_parse_json(text)
    if isinstance(data, (list, tuple)):
        out = []
        for item in data:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    start = int(item[0])
                    dur = int(item[1])
                    # Third element is the color vector
                    color_data = item[2]
                    if isinstance(color_data, (list, tuple)) and len(color_data) >= 3:
                        r, g, b = float(color_data[0]), float(color_data[1]), float(color_data[2])
                    else:
                        # Fallback: parse if it's a string representation
                        r, g, b = parse_vector3(color_data)
                    out.append((start, dur, (r, g, b)))
            except (ValueError, TypeError, IndexError):
                continue # Skip malformed items
        return sorted(out, key=lambda x: x[0])

    # Fallback: Legacy semicolon-separated format parsing
    txt = str(text).strip()
    if txt.startswith('{') and txt.endswith('}'):
        txt = txt[1:-1].strip()
    segments = []
    cur = ""
    depth = 0
    for ch in txt:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        cur += ch
        if depth == 0 and ch in ');':
            segments.append(cur.strip('();'))
            cur = ""
    if cur.strip('();'):
        segments.append(cur.strip('();'))

    out = []
    for seg in segments:
        try:
            parts = [p.strip() for p in seg.split(';') if p.strip()]
            if len(parts) >= 5: # start;dur;r;g;b
                start = int(parts[0])
                dur = int(parts[1])
                r = float(parts[2])
                g = float(parts[3])
                b = float(parts[4])
                out.append((start, dur, (r, g, b)))
        except (ValueError, IndexError):
            continue
    return sorted(out, key=lambda x: x[0])

def calculate_bezier_points(p0: Tuple[float, float, float], p1: Tuple[float, float, float],
                            pull_point: Tuple[float, float, float], pull_power: float,
                            num_samples: int = 20) -> List[Tuple[float, float, float]]:
    """
    Calculates points along a quadratic Bezier curve influenced by a pull point and power.

    Args:
        p0: Start point (x, y, z).
        p1: End point (x, y, z).
        pull_point: The point influencing the curve's shape.
        pull_power: The strength of the pull (0.0 to 1.0+).
        num_samples: Number of points to generate along the curve.

    Returns:
        A list of (x, y, z) tuples representing points on the curve.
    """
    if num_samples < 2:
        return [p0, p1]

    # Calculate the actual control point P_c based on pull_point and pull_power
    # P_c = P_pull + power * (P_pull - midpoint_of_P0_P1)
    # This creates a more intuitive "pull" effect.
    mid_p0_p1 = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0, (p0[2] + p1[2]) / 2.0)
    diff_x = pull_point[0] - mid_p0_p1[0]
    diff_y = pull_point[1] - mid_p0_p1[1]
    diff_z = pull_point[2] - mid_p0_p1[2]

    p_c_x = pull_point[0] + pull_power * diff_x
    p_c_y = pull_point[1] + pull_power * diff_y
    p_c_z = pull_point[2] + pull_power * diff_z
    p_c = (p_c_x, p_c_y, p_c_z)

    points = []
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0.0

        # Quadratic Bezier formula: B(t) = (1-t)^2 * P0 + 2*(1-t)*t * Pc + t^2 * P1
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p_c[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p_c[1] + t ** 2 * p1[1]
        z = (1 - t) ** 2 * p0[2] + 2 * (1 - t) * t * p_c[2] + t ** 2 * p1[2]

        points.append((x, y, z))

    return points

def validate_movements(movements):
    """Validate that movements do not overlap"""
    # Simplified for tuples: (start, duration, ...)
    sorted_movements = sorted(movements, key=lambda m: m[0])  # Sort by start time
    for i in range(1, len(sorted_movements)):
        prev_end = sorted_movements[i-1][0] + sorted_movements[i-1][1]
        if sorted_movements[i][0] < prev_end:
            raise ValueError(f"Overlapping movements detected at {sorted_movements[i][0]}ms")


class ZeroInterpreter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        # Fix: Set row_factory for named access (Issue #1)
        self.conn.row_factory = sqlite3.Row
        self.points: Dict[str, Dict[str, Any]] = {}
        self.lines: Dict[str, Dict[str, Any]] = {}
        self.shapes: Dict[str, Dict[str, Any]] = {}
        self.construct_the_model()

    def construct_the_model(self):
        """Loads all points, lines, and shapes from the database into memory."""
        cur = self.conn.cursor()
        # --- Load Points ---
        try:
            rows = cur.execute("SELECT coordinates, uuid, connected_points, movements FROM points").fetchall()
            for r in rows:
                try:
                    coords = parse_vector3(r["coordinates"])
                    uuid = str(r["uuid"])
                    conns = parse_uuid_list(r["connected_points"])
                    moves_str = r["movements"] if r["movements"] else ""
                    moves = parse_movements_field(moves_str)
                    # Validate movements don't overlap (as per old code logic)
                    if moves:
                        validate_movements(moves)

                    self.points[uuid] = {
                        "base": coords,
                        "connections": conns,
                        "movements": moves
                    }
                except Exception as e:
                    print(f"Warning: skipping malformed point {r.get('uuid', 'unknown')}: {e}")
        except sqlite3.OperationalError:
            print("Warning: points table not found or invalid structure")

        # --- Load Lines ---
        try:
            rows = cur.execute("SELECT uuid, endpoints, pull_point, pull_power, movements FROM lines").fetchall()
            for r in rows:
                try:
                    uuid = str(r["uuid"])
                    endpoints = parse_uuid_list(r["endpoints"])
                    if len(endpoints) != 2:
                        raise ValueError("Line must connect exactly two points")
                    pull_p_str = r["pull_point"]
                    pull_point = parse_vector3(pull_p_str) if pull_p_str else (0.0, 0.0, 0.0)
                    # Ensure pull_power is a float, defaulting to 1.0 if null/invalid
                    pull_power_raw = r["pull_power"]
                    if pull_power_raw is None:
                        pull_power = 1.0
                    else:
                        try:
                            pull_power = float(pull_power_raw)
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid pull_power '{pull_power_raw}' for line {uuid}, defaulting to 1.0")
                            pull_power = 1.0

                    moves_str = r["movements"] if r["movements"] else ""
                    moves = parse_line_changes_field(moves_str)
                     # Validate line changes don't overlap (as per old code logic)
                    if moves:
                        # Convert to format validate_movements expects (start, duration)
                        simplified_changes = [(ch[0], ch[1]) for ch in moves]
                        validate_movements(simplified_changes)

                    self.lines[uuid] = {
                        "endpoints": endpoints, # List of 2 UUIDs
                        "pull_point": pull_point,
                        "pull_power": pull_power,
                        "movements": moves # Renamed from 'changes' for consistency
                    }
                except Exception as e:
                    print(f"Warning: skipping malformed line {r.get('uuid', 'unknown')}: {e}")
        except sqlite3.OperationalError:
            print("Warning: lines table not found or invalid structure")

        # --- Load Shapes ---
        try:
            rows = cur.execute("SELECT uuid, point_uuids, line_uuids, color, movements FROM shapes").fetchall()
            for r in rows:
                try:
                    uuid = str(r["uuid"])
                    point_uuids = parse_uuid_list(r["point_uuids"])
                    line_uuids = parse_uuid_list(r["line_uuids"])
                    color_str = r["color"]
                    # FIXED: Ensure color is always a 3-tuple (as per old code)
                    try:
                        color = parse_vector3(color_str)
                    except (ValueError, TypeError):
                        color = (1.0, 1.0, 1.0) # Default white
                    moves_str = r["movements"] if r["movements"] else ""
                    moves = parse_color_changes(moves_str)
                    # Validate color changes don't overlap (as per old code logic)
                    if moves:
                        # Convert to format validate_movements expects (start, duration)
                        simplified_changes = [(ch[0], ch[1]) for ch in moves]
                        validate_movements(simplified_changes)

                    self.shapes[uuid] = {
                        "points": point_uuids,
                        "lines": line_uuids,
                        "color": color, # Renamed from 'fill_base' for consistency with old output
                        "movements": moves # Renamed from 'color_changes' for consistency
                    }
                except Exception as e:
                    print(f"Warning: skipping malformed shape {r.get('uuid', 'unknown')}: {e}")
        except sqlite3.OperationalError:
            print("Warning: shapes table not found or invalid structure")

        cur.close()

    def get_point_at_time(self, uuid: str, tick_ms: int) -> Tuple[float, float, float]:
        """Calculates the position of a point at a given time, considering movements.
        NOTE: Movements for a single point must not overlap (enforced by documentation).
        If they do overlap, behavior is undefined and the input should be considered invalid.
        """
        rec = self.points.get(uuid)
        if rec is None:
            raise KeyError(f"Point {uuid} not found")
        base = rec["base"]
        movements = rec["movements"] # Renamed from 'movements'
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
            t = (tick_ms - start) / dur
            # Linear interpolation
            current = lerp_vec(current, target, clamp(t)) # Use lerp_vec
        return current

    def get_line_at_time(self, uuid: str, tick_ms: int) -> Dict[str, Any]:
        """Calculates the state of a line at a given time, including curved points.
        NOTE: Line changes must not overlap (enforced by documentation).
        If they do overlap, behavior is undefined and the input should be considered invalid.
        """
        rec = self.lines.get(uuid)
        if rec is None:
            raise KeyError(f"Line {uuid} not found")

        ep1_uuid, ep2_uuid = rec["endpoints"]
        # Get dynamic positions of endpoints
        p1 = self.get_point_at_time(ep1_uuid, tick_ms)
        p2 = self.get_point_at_time(ep2_uuid, tick_ms)

        # Get pull point and power (these might have movements in the future, but not specified)
        pull_point_base = rec["pull_point"]
        pull_power_base = rec["pull_power"]

        # Handle movements for the line's pull point and power
        movements = rec["movements"] # Renamed from 'changes'
        current_pull_point = pull_point_base
        current_pull_power = pull_power_base
        for m in movements: # m = (start, duration, (x,y,z), power)
            start, dur, new_coords, new_power = m
            end = start + dur
            if tick_ms < start:
                continue
            if dur <= 0:
                current_pull_point = new_coords
                current_pull_power = new_power
                continue
            if tick_ms >= end:
                current_pull_point = new_coords
                current_pull_power = new_power
                continue
            progress = (tick_ms - start) / dur
            # Linear interpolation for pull point and power
            current_pull_point = lerp_vec(current_pull_point, new_coords, clamp(progress))
            current_pull_power = lerp(current_pull_power, new_power, clamp(progress))
            # Return immediately after finding the active movement (as per old logic)
            # This assumes movements are non-overlapping and sorted.
            break # Exit loop after processing the first matching movement

        # Sample the Bezier curve
        # Fix: Implement actual Bezier curve calculation (Issue #2)
        # Option A: Interpreter calculates points
        sampled_points = calculate_bezier_points(p1, p2, current_pull_point, current_pull_power, num_samples=20)

        return {
            "endpoints": (p1, p2),
            "pull_point": current_pull_point,
            "pull_power": current_pull_power,
            "sampled_points": sampled_points # New key for rendered points
        }

    def get_shape_at_time(self, uuid: str, tick_ms: int) -> Dict[str, Any]:
        """Calculates the state of a shape at a given time.
        NOTE: Color changes must not overlap (enforced by documentation).
        If they do overlap, behavior is undefined and the input should be considered invalid.
        """
        rec = self.shapes.get(uuid)
        if rec is None:
            raise KeyError(f"Shape {uuid} not found")

        base_color = rec["color"] # Renamed from 'fill_base'
        movements = rec["movements"] # Renamed from 'color_changes'
        current_color = base_color
        for m in movements: # m = (start, duration, (r,g,b))
            start, dur, target_color = m
            end = start + dur
            if tick_ms < start:
                continue
            if dur <= 0 or tick_ms >= end:
                current_color = target_color
                continue
            t = (tick_ms - start) / dur
            # Linear interpolation for color
            current_color = lerp_vec(current_color, target_color, clamp(t)) # Use lerp_vec
            # Return immediately after finding the active movement (as per old logic)
            # This assumes movements are non-overlapping and sorted.
            break # Exit loop after processing the first matching movement

        return {
            "point_uuids": rec["points"], # Keep original key names for output consistency
            "line_uuids": rec["lines"],   # Keep original key names for output consistency
            "color": current_color
        }

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
        # Updated query to match schema: id is AUTOINCREMENT, timestamp_ms is indexed
        cur.execute("""
            SELECT id, timestamp_ms, notes, durations, instrument_id 
            FROM music 
            WHERE timestamp_ms <= ? AND (timestamp_ms + CAST(json_extract(durations, '$[0]') AS INTEGER)) > ?
        """, (end_time_ms, start_time_ms))
        events = []
        for row in cur.fetchall():
            try:
                # Use row_factory for named access
                notes = json.loads(row["notes"])
                durations = json.loads(row["durations"])
                events.append({
                    "id": row["id"],
                    "timestamp_ms": row["timestamp_ms"],
                    "notes": notes,
                    "durations": durations,
                    "instrument_id": row["instrument_id"]
                })
            except Exception as e:
                print(f"Warning: skipping malformed music event at {row['timestamp_ms']}: {e}")
        cur.close()
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
        # Updated query to match schema: id is AUTOINCREMENT
        cur.execute("""
            SELECT id, sentence, start_time_ms, voice_id 
            FROM speech 
            WHERE start_time_ms >= ? AND start_time_ms <= ?
        """, (start_time_ms, end_time_ms))
        events = []
        for row in cur.fetchall():
            # Use row_factory for named access
            events.append({
                "id": row["id"],
                "sentence": row["sentence"],
                "start_time_ms": row["start_time_ms"],
                "voice_id": row["voice_id"] # Fixed key name from 'voice' to 'voice_id'
            })
        cur.close()
        return events

    def read_full(self, tick_in_ms: int) -> Dict[str, Any]:
        """
        Returns the full state of the scene at a given time tick.
        Includes calculated positions for points, sampled points for curved lines, and shape colors.
        """
        state = {
            "points": {},
            "lines": {}, # This will now contain 'sampled_points'
            "shapes": {},
            "tick": tick_in_ms
        }

        # Populate points
        for uuid in self.points:
            try:
                state["points"][uuid] = self.get_point_at_time(uuid, tick_in_ms)
            except KeyError:
                pass # Point not found, skip

        # Populate lines (with calculated curves)
        for uuid in self.lines:
            try:
                state["lines"][uuid] = self.get_line_at_time(uuid, tick_in_ms)
            except KeyError:
                pass # Line not found, skip

        # Populate shapes
        for uuid in self.shapes:
            try:
                state["shapes"][uuid] = self.get_shape_at_time(uuid, tick_in_ms)
            except KeyError:
                pass # Shape not found, skip

        return state

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()

# Example usage (if run as script)
if __name__ == "__main__":
    import sys
    import os
    DB_PATH = "graphics.db"
    if len(sys.argv) > 1:
        DB_PATH = sys.argv[1]
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file '{DB_PATH}' not found.")
        sys.exit(1)

    try:
        interpreter = ZeroInterpreter(DB_PATH)
        # Example: Read state at 1000ms
        state = interpreter.read_full(1000)
        print(json.dumps(state, indent=2)) # Pretty print the state
        interpreter.close()
    except Exception as e:
        print(f"Error initializing interpreter or reading state: {e}")
        sys.exit(1)
