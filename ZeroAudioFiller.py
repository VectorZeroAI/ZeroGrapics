#ZeroAudioFiller.py
"""ZeroAudioFiller.py - Fills out the audio tables in the database
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
        durations: List of durations corresponding to the notes
        instrument_id: Identifier for the instrument
    """
    if not notes or not durations or len(notes) != len(durations):
        print(f"Warning: Invalid music data at timestamp {timestamp_ms}. Skipping.")
        return

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO music (timestamp_ms, notes, durations, instrument_id) VALUES (?, ?, ?, ?)",
        (timestamp_ms, json.dumps(notes), json.dumps(durations), instrument_id)
    )
    conn.commit()

def insert_speech_entry(conn: sqlite3.Connection, sentence: str, start_time_ms: int, voice_id: str):
    """Insert a speech entry into the database

    Args:
        conn: SQLite database connection
        sentence: The sentence to be spoken
        start_time_ms: Timestamp when speech should start
        voice_id: Identifier for the voice
    """
    if not sentence:
        print(f"Warning: Empty sentence at time {start_time_ms}. Skipping.")
        return

    cur = conn.cursor()
    # Note: 'id' is AUTOINCREMENT, so it's omitted and will be generated automatically.
    cur.execute(
        "INSERT INTO speech (sentence, start_time_ms, voice_id) VALUES (?, ?, ?)",
        (sentence, start_time_ms, voice_id)
    )
    conn.commit()

def process_music_data(conn: sqlite3.Connection, data: List[Dict[str, Any]]):
    """Process and insert music data into the database

    Args:
        conn: SQLite database connection
        data: List of music event dictionaries
    """
    for item in data:
        timestamp_ms = int(item.get("timestamp_ms", 0))
        notes = item.get("notes", [])
        durations = item.get("durations", [])
        instrument_id = item.get("instrument_id", "default")

        # Validate notes and durations are lists
        if not isinstance(notes, list) or not isinstance(durations, list):
            print(f"Warning: Invalid format for notes/durations at timestamp {timestamp_ms}. Skipping.")
            continue

        # Ensure notes and durations are integers
        try:
            notes = [int(n) for n in notes]
            durations = [int(d) for d in durations]
        except (ValueError, TypeError):
            print(f"Warning: Non-integer values in notes/durations at timestamp {timestamp_ms}. Skipping.")
            continue

        insert_music_entry(conn, timestamp_ms, notes, durations, instrument_id)

def process_speech_data(conn: sqlite3.Connection, data: List[Dict[str, Any]]):
    """Process and insert speech data into the database

    Args:
        conn: SQLite database connection
        data: List of speech event dictionaries
    """
    for item in data:
        # Corrected query: Selecting 'id' is fine with AUTOINCREMENT, but ensure usage is correct downstream.
        # The original issue was conceptual, not with this specific query.
        sentence = item.get("sentence", "")
        start_time_ms = int(item.get("start_time_ms", 0))
        voice_id = item.get("voice_id", "default")
        insert_speech_entry(conn, sentence, start_time_ms, voice_id)

def parse_legacy_input(raw_input: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parses the legacy text format for audio data.

    Expected format:
    MUSIC:
    timestamp_ms1;[note1,duration1;note2,duration2;...];instrument_id
    ...
    SPEECH:
    start_time_ms1;sentence1;voice_id
    ...

    Args:
        raw_input: Raw string input in legacy format.

    Returns:
        A tuple containing lists of music and speech dictionaries.
    """
    music_data = []
    speech_data = []
    current_section = None

    for line_num, line in enumerate(raw_input.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line == "MUSIC:":
            current_section = "music"
            continue
        elif line == "SPEECH:":
            current_section = "speech"
            continue

        if current_section == "music":
            parts = line.split(";", 2)
            if len(parts) < 3:
                print(f"Warning (Line {line_num}): Invalid music format. Expected 3 parts. Skipping line.")
                continue
            
            try:
                timestamp_ms_raw, notes_durations_raw, instrument_id = parts
                timestamp_ms = int(timestamp_ms_raw.strip())
                
                # Process notes and durations
                notes = []
                durations = []
                if notes_durations_raw.startswith("[") and notes_durations_raw.endswith("]"):
                    nd_pairs = notes_durations_raw[1:-1].split(";")
                    for pair in nd_pairs:
                        if pair:
                            n_d = pair.split(",", 1)
                            if len(n_d) == 2:
                                # Corrected pull_power handling logic for legacy parser
                                # This one seems okay as is, but ensure consistency.
                                # power = float(power_raw) if str(power_raw).strip()!='' else 1.0
                                # If power_raw is "0", this correctly becomes 0.0.
                                note_str, dur_str = n_d
                                notes.append(int(note_str.strip()))
                                durations.append(int(dur_str.strip()))
                            else:
                                print(f"Warning (Line {line_num}): Invalid note,duration pair '{pair}'. Skipping pair.")
                
                music_data.append({
                    "timestamp_ms": timestamp_ms,
                    "notes": notes,
                    "durations": durations,
                    "instrument_id": instrument_id.strip()
                })
            except ValueError as e:
                print(f"Warning (Line {line_num}): Error parsing music line: {e}. Skipping line.")

        elif current_section == "speech":
            parts = line.split(";", 2)
            if len(parts) < 3:
                 print(f"Warning (Line {line_num}): Invalid speech format. Expected 3 parts. Skipping line.")
                 continue
            
            try:
                start_time_ms_raw, sentence, voice_id = parts
                start_time_ms = int(start_time_ms_raw.strip())
                speech_data.append({
                    "sentence": sentence.strip(),
                    "start_time_ms": start_time_ms,
                    "voice_id": voice_id.strip()
                })
            except ValueError as e:
                 print(f"Warning (Line {line_num}): Error parsing speech line: {e}. Skipping line.")
        else:
             print(f"Warning (Line {line_num}): Line outside of section or unknown section: {line}. Skipping.")

    return music_data, speech_data

def clear_audio_tables(conn: sqlite3.Connection):
    """Clear the audio tables before inserting new data"""
    cur = conn.cursor()
    cur.execute("DELETE FROM music;")
    cur.execute("DELETE FROM speech;")
    conn.commit()

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
        except IOError as e:
            print(f"Error reading file '{args.input}': {e}")
            conn.close()
            sys.exit(1)
    else:
        parser.print_help()
        conn.close()
        sys.exit(1)

    music_data = []
    speech_data = []

    # Parse input
    raw_input_stripped = raw_input.strip()
    if raw_input_stripped.startswith('[') or raw_input_stripped.startswith('{'):
        # Assume JSON
        try:
            data = json.loads(raw_input)
            if isinstance(data, dict):
                # If top-level is a dict, expect 'music' and 'speech' keys
                music_data = data.get("music", [])
                speech_data = data.get("speech", [])
            elif isinstance(data, list):
                # If top-level is a list, assume it's a list of events with a 'type' key
                for item in data:
                    if item.get("type") == "music":
                        music_data.append(item)
                    elif item.get("type") == "speech":
                        speech_data.append(item)
            else:
                print("Error: JSON root must be an object or array.")
                conn.close()
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            conn.close()
            sys.exit(1)
    else:
        # Assume legacy text format
        music_data, speech_data = parse_legacy_input(raw_input)

    # Process and insert data
    if music_data:
        process_music_data(conn, music_data)
        print(f"Inserted {len(music_data)} music entries.")
    else:
        print("No music data found to insert.")

    if speech_data:
        process_speech_data(conn, speech_data)
        print(f"Inserted {len(speech_data)} speech entries.")
    else:
        print("No speech data found to insert.")

    conn.close()
    print("Audio data filling complete.")

if __name__ == "__main__":
    main()
