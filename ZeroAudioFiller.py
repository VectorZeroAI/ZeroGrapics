"""
ZeroAudioFiller.py
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
    """
    Insert a music entry into the database
    
    Args:
        conn: SQLite database connection
        timestamp_ms: Timestamp in milliseconds
        notes: List of MIDI note numbers
        durations: List of note durations in milliseconds
        instrument_id: Instrument identifier
    """
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO music (timestamp_ms, notes, durations, instrument_id)
        VALUES (?, ?, ?, ?)
    """, (timestamp_ms, json.dumps(notes), json.dumps(durations), instrument_id))
    conn.commit()

def insert_speech_entry(conn: sqlite3.Connection, sentence: str, 
                       start_time_ms: int, voice_id: str):
    """
    Insert a speech entry into the database
    
    Args:
        conn: SQLite database connection
        sentence: Text to speak
        start_time_ms: When the speech should start (in milliseconds)
        voice_id: Voice identifier
    """
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO speech (sentence, start_time_ms, voice_id)
        VALUES (?, ?, ?)
    """, (sentence, start_time_ms, voice_id))
    conn.commit()

def process_music_data(conn: sqlite3.Connection, music_data: Dict[str, Any]):
    """
    Process music data from a JSON object
    
    Args:
        conn: SQLite database connection
        music_data: Dictionary containing music data
    
    Expected format:
    {
        "music": [
            {
                "time_ms": 100,
                "notes": [60, 64, 67],
                "durations": [500, 500, 500],
                "instrument": "piano"
            },
            ...
        ]
    }
    """
    music_entries = music_data.get("music", [])
    for entry in music_entries:
        time_ms = entry.get("time_ms", 0)
        notes = entry.get("notes", [])
        durations = entry.get("durations", [])
        instrument = entry.get("instrument", "piano")
        
        # Ensure notes and durations are lists of numbers
        notes = [int(n) for n in notes if isinstance(n, (int, float))]
        durations = [int(d) for d in durations if isinstance(d, (int, float))]
        
        # Only insert if we have valid data
        if notes and durations and len(notes) == len(durations):
            insert_music_entry(conn, time_ms, notes, durations, instrument)

def process_speech_data(conn: sqlite3.Connection, speech_data: Dict[str, Any]):
    """
    Process speech data from a JSON object
    
    Args:
        conn: SQLite database connection
        speech_data: Dictionary containing speech data
    
    Expected format:
    {
        "speech": [
            {
                "sentence": "Hello, this is a test.",
                "start_time_ms": 2000,
                "voice": "female_english"
            },
            ...
        ]
    }
    """
    speech_entries = speech_data.get("speech", [])
    for entry in speech_entries:
        sentence = entry.get("sentence", "")
        start_time_ms = entry.get("start_time_ms", 0)
        voice = entry.get("voice", "default")
        
        # Only insert if we have a non-empty sentence
        if sentence.strip():
            insert_speech_entry(conn, sentence, start_time_ms, voice)

def process_json_input(conn: sqlite3.Connection, data: Dict[str, Any]):
    """
    Process a JSON input file containing audio data
    
    Args:
        conn: SQLite database connection
        data: Parsed JSON data
    """
    # Process music data
    process_music_data(conn, data)
    
    # Process speech data
    process_speech_data(conn, data)

def process_legacy_text(conn: sqlite3.Connection, raw_text: str):
    """
    Process legacy text format for audio data
    
    Expected format:
    MUSIC:
    100:{60;64;67};{500;500;500};piano
    200:{62;65;69};{300;300;300};guitar
    
    SPEECH:
    2000:Hello, this is a test.:female_english
    5000:Another sentence to speak.:male_english
    """
    # Extract MUSIC block
    music_block = _extract_block(raw_text, "MUSIC")
    if music_block:
        _process_music_block(conn, music_block)
    
    # Extract SPEECH block
    speech_block = _extract_block(raw_text, "SPEECH")
    if speech_block:
        _process_speech_block(conn, speech_block)

def _extract_block(text: str, block_name: str) -> str:
    """
    Extract a block of text between headers
    
    Args:
        text: Input text
        block_name: Name of the block to extract
    
    Returns:
        Extracted block text or empty string if not found
    """
    start_marker = f"{block_name}:"
    end_marker = ":END"
    
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    
    start_idx += len(start_marker)
    
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        end_idx = len(text)
    
    return text[start_idx:end_idx].strip()

def _process_music_block(conn: sqlite3.Connection, block: str):
    """
    Process a MUSIC block in legacy format
    
    Format: time:{notes};{durations};instrument
    Example: 100:{60;64;67};{500;500;500};piano
    """
    lines = block.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
            
        try:
            time_ms = int(parts[0].strip())
            data = parts[1].strip()
            
            # Split the data part
            data_parts = data.split(";", 3)
            if len(data_parts) < 3:
                continue
                
            # Parse notes
            notes_str = data_parts[0].strip("{}")
            notes = [int(n.strip()) for n in notes_str.split(";") if n.strip().isdigit()]
            
            # Parse durations
            durations_str = data_parts[1].strip("{}")
            durations = [int(d.strip()) for d in durations_str.split(";") if d.strip().isdigit()]
            
            # Get instrument
            instrument = data_parts[2].strip()
            
            # Insert into database
            if notes and durations and len(notes) == len(durations):
                insert_music_entry(conn, time_ms, notes, durations, instrument)
                
        except (ValueError, IndexError):
            continue

def _process_speech_block(conn: sqlite3.Connection, block: str):
    """
    Process a SPEECH block in legacy format
    
    Format: time:sentence:voice
    Example: 2000:Hello, this is a test.:female_english
    """
    lines = block.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
            
        try:
            time_ms = int(parts[0].strip())
            sentence = parts[1].strip()
            voice = parts[2].strip()
            
            # Insert into database
            insert_speech_entry(conn, sentence, time_ms, voice)
            
        except (ValueError, IndexError):
            continue

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
        except FileNotFoundError:
            print(f"Error: Input file '{args.input}' not found.")
            sys.exit(1)
    else:
        print("Error: No input provided. Use --stdin or specify an input file.")
        parser.print_help()
        sys.exit(1)
    
    # Try to parse as JSON
    try:
        data = json.loads(raw_input)
        process_json_input(conn, data)
        print("Processed audio data from JSON input.")
    except json.JSONDecodeError:
        # Not JSON, treat as legacy text format
        process_legacy_text(conn, raw_input)
        print("Processed audio data from legacy text input.")
    
    conn.close()
    print("Audio data insertion complete.")

if __name__ == "__main__":
    main()