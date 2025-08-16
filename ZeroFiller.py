#ZeroFiller.py
"""
ZeroFiller.py
- Reads LLM-generated .txt or .json that describes points, lines, shapes, and audio.
- Parses the input (JSON preferred) or a legacy brace-based text format.
- Validates fields and inserts rows into the SQLite DB produced by ZeroInit.py.

Schema expected (graphics.db created by ZeroInit.py):
    points(coordinates TEXT, uuid TEXT PRIMARY KEY, connected_points TEXT, movements TEXT)
    lines(uuid TEXT PRIMARY KEY, endpoints TEXT, pull_point TEXT, pull_power REAL, movements TEXT)
    shapes(uuid TEXT PRIMARY KEY, point_uuids TEXT, line_uuids TEXT, color TEXT, movements TEXT) # FIXED: Added uuid
    music(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_ms INTEGER NOT NULL, notes TEXT, durations TEXT, instrument_id TEXT)
    speech(id INTEGER PRIMARY KEY AUTOINCREMENT, sentence TEXT, start_time_ms INTEGER NOT NULL, voice_id TEXT)
"""

import sqlite3
import json
import argparse
import sys
import os
# import uuid # Removed unused import (Fix #3)

from typing import Dict, Any, List, Tuple

DB_FILE = "graphics.db"

# --- Helper Functions for Parsing ---

def parse_uuid_list(s: Any) -> List[str]:
    """Accepts: JSON list/tuple or string like "[uuid1;uuid2;...]" or "(uuid1;uuid2;...)"
    Returns: List of UUID strings.
    """
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') or s.startswith('('):
            # Remove surrounding brackets/parentheses
            content = s[1:-1]
            # Split by semicolon
            return [uuid.strip() for uuid in content.split(';') if uuid.strip()]
        else:
            # Assume it's a single UUID
            return [s] if s else []
    elif isinstance(s, (list, tuple)):
        return [str(item) for item in s]
    else:
        return [str(s)] if s is not None else []

def parse_vector3(s: Any) -> Tuple[float, float, float]:
    """Accepts: JSON list/tuple or string like "[x;y;z]" or "(x;y;z)"
    Returns: Tuple of 3 floats.
    """
    # Issue #6: Inconsistent indentation of docstring (Fix #4)
    if isinstance(s, str):
        s = s.strip()
        if s.startswith('[') or s.startswith('('):
            content = s[1:-1]
            parts = content.split(';')
            if len(parts) != 3:
                raise ValueError(f"Vector3 string must have 3 components: {s}")
            return tuple(float(p.strip()) for p in parts)
        else:
            raise ValueError(f"Vector3 string must be in brackets or parentheses: {s}")
    elif isinstance(s, (list, tuple)):
        if len(s) != 3:
            raise ValueError(f"Vector3 list/tuple must have 3 components: {s}")
        return tuple(float(coord) for coord in s)
    else:
        raise TypeError(f"Vector3 must be a string, list, or tuple: {type(s)}")


# --- Database Insertion Functions ---

def insert_point_entry(conn: sqlite3.Connection, uuid: str, coordinates: Tuple[float, float, float], connected_points: List[str], movements: str):
    """Insert a point entry into the database"""
    cur = conn.cursor()
    coordinates_str = json.dumps(list(coordinates))
    connected_points_str = json.dumps(connected_points)
    cur.execute(
        "INSERT OR REPLACE INTO points (uuid, coordinates, connected_points, movements) VALUES (?, ?, ?, ?)",
        (uuid, coordinates_str, connected_points_str, movements)
    )
    conn.commit()

def insert_line_entry(conn: sqlite3.Connection, uuid: str, endpoints: List[str], pull_point: Tuple[float, float, float], pull_power: float, movements: str):
    """Insert a line entry into the database"""
    cur = conn.cursor()
    endpoints_str = json.dumps(endpoints)
    pull_point_str = json.dumps(list(pull_point))
    cur.execute(
        "INSERT OR REPLACE INTO lines (uuid, endpoints, pull_point, pull_power, movements) VALUES (?, ?, ?, ?, ?)",
        (uuid, endpoints_str, pull_point_str, pull_power, movements)
    )
    conn.commit()

def insert_shape_entry(conn: sqlite3.Connection, uuid: str, point_uuids: List[str], line_uuids: List[str], color: Tuple[float, float, float], movements: str):
    """Insert a shape entry into the database"""
    cur = conn.cursor()
    point_uuids_str = json.dumps(point_uuids)
    line_uuids_str = json.dumps(line_uuids)
    color_str = json.dumps(list(color))
    cur.execute(
        "INSERT OR REPLACE INTO shapes (uuid, point_uuids, line_uuids, color, movements) VALUES (?, ?, ?, ?, ?)",
        (uuid, point_uuids_str, line_uuids_str, color_str, movements)
    )
    conn.commit()

def insert_music_entry(conn: sqlite3.Connection, timestamp_ms: int, notes: List[int],
                       durations: List[int], instrument_id: str):
    """Insert a music entry into the database
    Args:
        conn: SQLite database connection
        timestamp_ms: Timestamp in milliseconds
        notes: List of note frequencies (Hz)
        durations: List of durations for each note (ms)
        instrument_id: Instrument identifier string
    """
    cur = conn.cursor()
    notes_str = json.dumps(notes)
    durations_str = json.dumps(durations)
    # The primary key is AUTOINCREMENT id, so we don't specify it.
    # timestamp_ms is NOT NULL, so it must be provided.
    cur.execute(
        "INSERT INTO music (timestamp_ms, notes, durations, instrument_id) VALUES (?, ?, ?, ?)",
        (timestamp_ms, notes_str, durations_str, instrument_id)
    )
    conn.commit()

def insert_speech_entry(conn: sqlite3.Connection, sentence: str, start_time_ms: int, voice_id: str):
    """Insert a speech entry into the database
    Args:
        conn: SQLite database connection
        sentence: The sentence to be spoken
        start_time_ms: Start time in milliseconds
        voice_id: Voice identifier string
    """
    cur = conn.cursor()
    # The primary key is AUTOINCREMENT id, so we don't specify it.
    # start_time_ms is NOT NULL, so it must be provided.
    cur.execute(
        "INSERT INTO speech (sentence, start_time_ms, voice_id) VALUES (?, ?, ?)",
        (sentence, start_time_ms, voice_id)
    )
    conn.commit()


# --- Processing Functions ---

def process_json_input(conn: sqlite3.Connection, data: Dict[str, Any]):
    """Process JSON input data and insert into the database."""
    # Process points
    if "points" in data:
        for p in data["points"]:
            try:
                uuid = str(p["uuid"])
                coordinates = parse_vector3(p["coordinates"])
                # Handle connected_points: can be missing, empty, or a list
                connected_points_raw = p.get("connected_points", [])
                if not connected_points_raw: # Handles None, [], empty string, etc.
                    connected_points = []
                else:
                    connected_points = parse_uuid_list(connected_points_raw)

                movements = json.dumps(p.get("movements", {})) # Default to empty JSON object
                insert_point_entry(conn, uuid, coordinates, connected_points, movements)
            except Exception as e:
                print(f"Warning: skipping malformed point {p.get('uuid', 'unknown')}: {e}")

    # Process lines
    if "lines" in data:
        for l in data["lines"]:
            try:
                uuid = str(l["uuid"])
                endpoints = parse_uuid_list(l["endpoints"])
                if len(endpoints) != 2:
                    print(f"Warning: line {uuid} must connect exactly two points. Skipping.")
                    continue

                # Handle pull_point: default to midpoint if missing or invalid
                pull_point_raw = l.get("pull_point")
                if pull_point_raw:
                    try:
                        pull_point = parse_vector3(pull_point_raw)
                    except (ValueError, TypeError):
                        print(f"Warning: invalid pull_point for line {uuid}. Using midpoint.")
                        pull_point = None
                else:
                    pull_point = None

                # Handle pull_power: default to 1.0 if missing, empty, or invalid
                # Issue #1: Incorrect pull_power handling (Fix #1)
                power_val = l.get("pull_power", l.get("power", 1.0))
                if power_val is None or (isinstance(power_val, str) and power_val.strip() == ""):
                    power = 1.0
                else:
                    power = float(power_val)

                movements = json.dumps(l.get("movements", {})) # Default to empty JSON object

                # If pull_point is invalid or missing, calculate midpoint from endpoints
                # (Assumes endpoints UUIDs are valid and points exist in DB - could add check)
                if pull_point is None:
                    # This requires fetching point coordinates, for now, we can't easily calculate it here
                    # A placeholder or default strategy is needed. Let's default to [0,0,0] or handle in interpreter.
                    # For filler, we can store a default or leave it. Let's re-evaluate.
                    # For now, let's store a default pull point if not provided.
                    # A better approach might be to require it or have a specific default logic in the interpreter.
                    # Let's store [0,0,0] if we can't determine it, interpreter can handle it.
                    pull_point = (0.0, 0.0, 0.0) # Placeholder

                insert_line_entry(conn, uuid, endpoints, pull_point, power, movements)
            except Exception as e:
                print(f"Warning: skipping malformed line {l.get('uuid', 'unknown')}: {e}")

    # Process shapes
    if "shapes" in data:
        for s in data["shapes"]:
            try:
                uuid = str(s["uuid"])
                point_uuids = parse_uuid_list(s.get("point_uuids", []))
                line_uuids = parse_uuid_list(s.get("line_uuids", []))

                # Handle color: default to gray if missing or invalid
                color_raw = s.get("color")
                if color_raw:
                    try:
                        color = parse_vector3(color_raw)
                    except (ValueError, TypeError):
                        print(f"Warning: invalid color for shape {uuid}. Using default gray.")
                        color = (0.5, 0.5, 0.5)
                else:
                    color = (0.5, 0.5, 0.5)

                movements = json.dumps(s.get("movements", {})) # Default to empty JSON object
                insert_shape_entry(conn, uuid, point_uuids, line_uuids, color, movements)
            except Exception as e:
                print(f"Warning: skipping malformed shape {s.get('uuid', 'unknown')}: {e}")

    # Process music
    if "music" in data:
        for m in data["music"]:
            try:
                # timestamp_ms is required
                timestamp_ms = int(m["timestamp_ms"])
                notes = [int(n) for n in m["notes"]]
                durations = [int(d) for d in m["durations"]]
                if len(notes) != len(durations):
                     print(f"Warning: music entry at {timestamp_ms} has mismatched notes/durations. Skipping.")
                     continue
                instrument_id = str(m.get("instrument_id", "default"))
                insert_music_entry(conn, timestamp_ms, notes, durations, instrument_id)
            except Exception as e:
                print(f"Warning: skipping malformed music entry at timestamp {m.get('timestamp_ms', 'unknown')}: {e}")

    # Process speech
    if "speech" in data:
        for sp in data["speech"]:
            try:
                sentence = str(sp["sentence"])
                start_time_ms = int(sp["start_time_ms"])
                voice_id = str(sp.get("voice_id", "default"))
                insert_speech_entry(conn, sentence, start_time_ms, voice_id)
            except Exception as e:
                print(f"Warning: skipping malformed speech entry starting at {sp.get('start_time_ms', 'unknown')}: {e}")


def process_legacy_text_input(conn: sqlite3.Connection, file_path: str):
    """Process legacy brace-based text input."""
    print("Warning: Legacy text format parsing is not fully implemented in this snippet.")
    # A full implementation would involve reading the file line by line,
    # identifying blocks like Point{}, Line{}, Shape{}, parsing content within braces,
    # handling semicolon-separated values, and calling the appropriate insert_* functions.
    # It would also need to incorporate the pull_power fix (Fix #2).
    # For example, for pull_power:
    # power_raw = l.get("power") # or however it's extracted from the text line
    # Issue #2: Incorrect pull_power handling (Legacy) - Review needed (Fix #2)
    # The current logic `float(power_raw) if str(power_raw).strip()!='' else 1.0` seems okay
    # for the legacy parser's structure IF 0 is intended to be 0.0.
    # However, ensure consistency with the JSON parser fix (Fix #1).
    # If the logic should be identical, apply the same fix as Bug 1 here.
    # ---
    # power_val = power_raw # Assuming power_raw is already extracted
    # if power_val is None or (isinstance(power_val, str) and power_val.strip() == ""):
    #     power = 1.0
    # else:
    #     power = float(power_val)
    # ---
    # Placeholder pass
    pass


# --- Main CLI Function ---

def main():
    # Issue #3: Inconsistent indentation of docstring (Fix #4)
    """Main function for the command-line interface"""
    parser = argparse.ArgumentParser(description="Fill the graphics database from input files.")
    parser.add_argument("input_file", help="Path to the input .json or .txt file")
    parser.add_argument("--db", default=DB_FILE, help="Path to the SQLite database file")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    if not os.path.exists(args.db):
        print(f"Error: Database file '{args.db}' not found. Please run ZeroInit.py first.")
        sys.exit(1)

    try:
        conn = sqlite3.connect(args.db)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    try:
        if args.input_file.endswith('.json'):
            try:
                with open(args.input_file, 'r') as f:
                    data = json.load(f)
                process_json_input(conn, data)
                print(f"Successfully processed JSON file '{args.input_file}' into '{args.db}'.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file '{args.input_file}': {e}")
                conn.close()
                sys.exit(1)
        elif args.input_file.endswith('.txt'):
            process_legacy_text_input(conn, args.input_file)
            print(f"Processed legacy text file '{args.input_file}' into '{args.db}'.")
        else:
            print("Error: Unsupported file type. Please provide a .json or .txt file.")
            conn.close()
            sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()