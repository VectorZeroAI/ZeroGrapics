#ZeroAudioFiller.py
"""ZeroAudioFiller.py
- Fills out the audio tables in the database
- Parses input data for music and speech
- Handles both JSON and legacy text formats
"""

import sqlite3
import json
import argparse
import sys
import os
from typing import Dict, Any, List, Tuple

DB_FILE = "graphics.db"

def insert_music_entry(conn: sqlite3.Connection, timestamp_ms: int, notes: List[int],
                       durations: List[int], instrument_id: str):
    """Insert a music entry into the database

    Args:
        conn: SQLite database connection
        timestamp_ms: Timestamp in milliseconds
        notes: List of MIDI note numbers
        durations: List of durations corresponding to notes
        instrument_id: Instrument identifier
    """
    if not notes or not durations or len(notes) != len(durations):
        print(f"Warning: Invalid music data at {timestamp_ms}ms. Skipping.")
        return

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO music (timestamp_ms, notes, durations, instrument_id) VALUES (?, ?, ?, ?)",
        (timestamp_ms, json.dumps(notes), json.dumps(durations), instrument_id)
    )
    conn.commit()

def insert_speech_entry(conn: sqlite3.Connection, sentence: str,
                        start_time_ms: int, voice_id: str):
    """Insert a speech entry into the database

    Args:
        conn: SQLite database connection
        sentence: The sentence to be spoken
        start_time_ms: Start time in milliseconds
        voice_id: Voice identifier
    """
    if not sentence.strip():
        print(f"Warning: Empty sentence at {start_time_ms}ms. Skipping.")
        return

    cur = conn.cursor()
    # Let the database handle the auto-incrementing 'id'
    cur.execute(
        "INSERT INTO speech (sentence, start_time_ms, voice_id) VALUES (?, ?, ?)",
        (sentence, start_time_ms, voice_id)
    )
    conn.commit()

def clear_audio_tables(conn: sqlite3.Connection):
    """Clear the audio tables before inserting new data"""
    cur = conn.cursor()
    cur.execute("DELETE FROM music;")
    cur.execute("DELETE FROM speech;")
    conn.commit()

def process_music_data(conn: sqlite3.Connection, music_data: List[Dict[str, Any]]):
    """Process and insert music data"""
    for entry in music_data:
        timestamp_ms = entry.get("timestamp_ms")
        notes = entry.get("notes", [])
        durations = entry.get("durations", [])
        instrument_id = entry.get("instrument_id")

        if timestamp_ms is None or instrument_id is None:
            print(f"Warning: Music entry missing required fields (timestamp_ms, instrument_id). Skipping entry: {entry}")
            continue

        insert_music_entry(conn, timestamp_ms, notes, durations, instrument_id)

def process_speech_data(conn: sqlite3.Connection, speech_data: List[Dict[str, Any]]):
    """Process and insert speech data"""
    for entry in speech_data:
        sentence = entry.get("sentence")
        start_time_ms = entry.get("start_time_ms")
        voice_id = entry.get("voice_id")

        if sentence is None or start_time_ms is None or voice_id is None:
            print(f"Warning: Speech entry missing required fields (sentence, start_time_ms, voice_id). Skipping entry: {entry}")
            continue

        insert_speech_entry(conn, sentence, start_time_ms, voice_id)

def parse_json_input(raw_input: str, conn: sqlite3.Connection):
    """Parse JSON input and fill database tables"""
    try:
        data = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON input: {e}")
        sys.exit(1)

    if not isinstance(data, dict):
        print("Error: JSON root must be an object.")
        sys.exit(1)

    # Clear tables if data is present in the input
    if "music" in data or "speech" in data:
        clear_audio_tables(conn)
        print("Cleared existing audio tables.")

    # Process music data
    if "music" in data:
        if not isinstance(data["music"], list):
            print("Error: 'music' key must contain a list of entries.")
            sys.exit(1)
        process_music_data(conn, data["music"])

    # Process speech data
    if "speech" in data:
        if not isinstance(data["speech"], list):
            print("Error: 'speech' key must contain a list of entries.")
            sys.exit(1)
        process_speech_data(conn, data["speech"])

def main():
    """Main function for the command-line interface"""
    parser = argparse.ArgumentParser(description="ZeroAudioFiller - Fill audio tables in the database")
    parser.add_argument("input", nargs="?", help="Input file (JSON or text). Use --stdin to read from stdin.")
    parser.add_argument("--stdin", action="store_true", help="Read input from STDIN")
    parser.add_argument("--db", default=DB_FILE, help="Path to SQLite DB (default graphics.db)")
    parser.add_argument("--clear", action="store_true", help="Clear audio tables before inserting")

    args = parser.parse_args()

    # Check if database exists
    if not os.path.exists(args.db):
        print(f"Error: Database file '{args.db}' not found.")
        print("Run ZeroInit.py to create the database first.")
        sys.exit(1)

    # Connect to database
    conn = sqlite3.connect(args.db)

    # Clear tables if requested
    if args.clear:
        clear_audio_tables(conn)
        print("Cleared existing audio tables.")

    # Read input
    if args.stdin:
        raw_input = sys.stdin.read()
    elif args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                raw_input = f.read()
        except FileNotFoundError:
            print(f"Error: Input file '{args.input}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Process input
    parse_json_input(raw_input, conn)
    conn.close()
    print("Audio data successfully inserted into the database.")

if __name__ == "__main__":
    main()
