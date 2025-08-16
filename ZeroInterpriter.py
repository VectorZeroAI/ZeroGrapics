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
from typing import Any, Dict, List, Tuple, Union

# --- Utility Functions ---

def parse_vector3(s: Any) -> Tuple[float, float, float]:
    """Parses a string like '{1.0;2.0;3.0}' into a tuple (1.0, 2.0, 3.0).
    Handles strings, tuples, lists, and returns (0,0,0) on error.
    """
    if isinstance(s, (tuple, list)) and len(s) == 3:
        try:
            return tuple(float(x) for x in s)
        except (ValueError, TypeError):
            pass
    if isinstance(s, str):
        try:
            parts = s.strip('{}').split(';')
            if len(parts) == 3:
                return tuple(float(p) for p in parts)
        except (ValueError, AttributeError):
            pass
    return (0.0, 0.0, 0.0)

def parse_uuid_list(s: Any) -> List[str]:
    """Parses a string like '{uuid1;uuid2}' into a list ['uuid1', 'uuid2'].
    Handles strings, lists, and returns [] on error.
    """
    if isinstance(s, list):
        return [str(item) for item in s]
    if isinstance(s, str):
        try:
            stripped = s.strip('{}')
            if not stripped:
                return []
            return [uuid.strip() for uuid in stripped.split(';') if uuid.strip()]
        except AttributeError:
            pass
    return []

def parse_movements(movements_str: str) -> List[Tuple[int, int, Tuple[float, float, float]]]:
    """Parses the movements string format into a list of (start, duration, target) tuples."""
    movements = []
    if not isinstance(movements_str, str) or not movements_str.strip():
        return movements
    try:
        # Assuming format like {(start;duration;[x;y;z]);...}
        main_parts = movements_str.strip('{}').split(');')
        for part in main_parts:
            if not part.strip():
                continue
            inner_part = part.strip('()')
            segments = inner_part.split(';')
            if len(segments) == 3:
                start = int(segments[0])
                duration = int(segments[1])
                target_vec_str = segments[2]
                target = parse_vector3(target_vec_str)
                movements.append((start, duration, target))
    except (ValueError, IndexError, AttributeError) as e:
        print(f"Error parsing movements string '{movements_str}': {e}")
    return movements

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
                    moves = parse_movements(moves_str)
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
                    moves = parse_movements(moves_str)
                    self.lines[uuid] = {
                        "endpoints": endpoints, # List of 2 UUIDs
                        "pull_point": pull_point,
                        "pull_power": pull_power,
                        "movements": moves
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
                    color = parse_vector3(color_str) if color_str else (1.0, 1.0, 1.0) # Default white
                    moves_str = r["movements"] if r["movements"] else ""
                    moves = parse_movements(moves_str)
                    self.shapes[uuid] = {
                        "points": point_uuids,
                        "lines": line_uuids,
                        "color": color,
                        "movements": moves
                    }
                except Exception as e:
                    print(f"Warning: skipping malformed shape {r.get('uuid', 'unknown')}: {e}")
        except sqlite3.OperationalError:
            print("Warning: shapes table not found or invalid structure")

        cur.close()

    def get_point_at_time(self, uuid: str, tick_ms: int) -> Tuple[float, float, float]:
        """Calculates the position of a point at a given time, considering movements."""
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
            t = (tick_ms - start) / dur
            # Linear interpolation
            current = (
                current[0] * (1 - t) + target[0] * t,
                current[1] * (1 - t) + target[1] * t,
                current[2] * (1 - t) + target[2] * t
            )
        return current

    def get_line_at_time(self, uuid: str, tick_ms: int) -> Dict[str, Any]:
        """Calculates the state of a line at a given time, including curved points."""
        rec = self.lines.get(uuid)
        if rec is None:
            raise KeyError(f"Line {uuid} not found")

        ep1_uuid, ep2_uuid = rec["endpoints"]
        # Get dynamic positions of endpoints
        p1 = self.get_point_at_time(ep1_uuid, tick_ms)
        p2 = self.get_point_at_time(ep2_uuid, tick_ms)

        # Get pull point and power (these might have movements in the future, but not specified)
        pull_point_base = rec["pull_point"]
        pull_power = rec["pull_power"]

        # Handle movements for the line's pull point if needed (currently not in schema)
        # For now, assume pull point/base is static, or its movements are handled differently.
        # If movements for pull point are added, they would be processed here similar to points.

        # Sample the Bezier curve
        # Fix: Implement actual Bezier curve calculation (Issue #2)
        # Option A: Interpreter calculates points
        sampled_points = calculate_bezier_points(p1, p2, pull_point_base, pull_power, num_samples=20)

        return {
            "endpoints": (p1, p2),
            "pull_point": pull_point_base, # Or its dynamic position if it moves
            "pull_power": pull_power,
            "sampled_points": sampled_points # New key for rendered points
        }

    def get_shape_at_time(self, uuid: str, tick_ms: int) -> Dict[str, Any]:
        """Calculates the state of a shape at a given time."""
        rec = self.shapes.get(uuid)
        if rec is None:
            raise KeyError(f"Shape {uuid} not found")

        base_color = rec["color"]
        movements = rec["movements"]
        current_color = base_color
        for m in movements:
            start, dur, target_color = m
            end = start + dur
            if tick_ms < start:
                continue
            if dur <= 0 or tick_ms >= end:
                current_color = target_color
                continue
            t = (tick_ms - start) / dur
            current_color = (
                current_color[0] * (1 - t) + target_color[0] * t,
                current_color[1] * (1 - t) + target_color[1] * t,
                current_color[2] * (1 - t) + target_color[2] * t
            )

        return {
            "point_uuids": rec["points"],
            "line_uuids": rec["lines"],
            "color": current_color
        }


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
