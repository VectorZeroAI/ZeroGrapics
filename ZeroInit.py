#ZeroInit.py
import sqlite3
import os
import json
import uuid

DB_FILE = "graphics.db"

def create_database(db_path=DB_FILE, overwrite=False):
    # Remove DB if overwriting
    if overwrite and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Overwriting existing database: {db_path}")

    # Connect to DB (creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create Table 1: points
    # Schema: coordinates TEXT, uuid TEXT PRIMARY KEY, connected_points TEXT, movements TEXT
    cur.execute("""
        CREATE TABLE IF NOT EXISTS points (
            coordinates TEXT NOT NULL,
            uuid TEXT PRIMARY KEY,
            connected_points TEXT,
            movements TEXT
        )
    """)

    # Create Table 2: lines
    # Schema: uuid TEXT PRIMARY KEY, endpoints TEXT, pull_point TEXT, pull_power REAL, movements TEXT
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lines (
            uuid TEXT PRIMARY KEY,
            endpoints TEXT NOT NULL,
            pull_point TEXT,
            pull_power REAL,
            movements TEXT
        )
    """)

    # Create Table 3: shapes
    # Schema: uuid TEXT PRIMARY KEY, point_uuids TEXT, line_uuids TEXT, color TEXT, movements TEXT
    cur.execute("""
        CREATE TABLE IF NOT EXISTS shapes (
            uuid TEXT PRIMARY KEY,
            point_uuids TEXT,
            line_uuids TEXT,
            color TEXT,
            movements TEXT
        )
    """)

    # Create Table 4: music (for audio support)
    # Schema: id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_ms INTEGER NOT NULL,
    #         notes TEXT NOT NULL, durations TEXT NOT NULL, instrument_id TEXT NOT NULL
    # Note: Storing lists as TEXT (JSON strings) is a simple approach.
    #       A more normalized structure (e.g., a separate notes table) is often better for complex queries.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS music (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            notes TEXT NOT NULL,        -- JSON list of note frequencies (e.g., "[440, 523]")
            durations TEXT NOT NULL,    -- JSON list of durations in ms (e.g., "[500, 250]")
            instrument_id TEXT NOT NULL -- Identifier for the instrument sound to use
        )
    """)

    # Create Table 5: speech (for audio support)
    # Schema: id INTEGER PRIMARY KEY AUTOINCREMENT, sentence TEXT NOT NULL,
    #         start_time_ms INTEGER NOT NULL, voice_id TEXT NOT NULL
    cur.execute("""
        CREATE TABLE IF NOT EXISTS speech (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL,
            start_time_ms INTEGER NOT NULL,
            voice_id TEXT NOT NULL -- Identifier for the TTS voice to use
        )
    """)

    # Commit changes
    conn.commit()
    conn.close()
    print(f"Database '{db_path}' created/initialized successfully.")

def seed_example_data(db_path=DB_FILE):
    """Optional: Inserts one sample row per table for debugging."""
    if not os.path.exists(db_path):
        print(f"Database '{db_path}' not found. Run create_database first.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Sample UUIDs
    point_uuid = str(uuid.uuid4())
    line_uuid = str(uuid.uuid4())
    shape_uuid = str(uuid.uuid4())

    # Insert sample point
    cur.execute("""
        INSERT INTO points (coordinates, uuid, connected_points, movements)
        VALUES (?, ?, ?, ?)
    """, (
        json.dumps([1.0, 2.0, 3.0]),  # coordinates
        point_uuid,                   # uuid
        json.dumps([]),              # connected_points
        json.dumps({})               # movements
    ))

    # Insert sample line
    cur.execute("""
        INSERT INTO lines (uuid, endpoints, pull_point, pull_power, movements)
        VALUES (?, ?, ?, ?, ?)
    """, (
        line_uuid,                            # uuid
        json.dumps([[0, 0, 0], [1, 1, 1]]), # endpoints
        json.dumps([0.5, 0.5, 0.5]),       # pull_point
        0.5,                               # pull_power
        json.dumps({})                     # movements
    ))

    # Insert sample shape
    cur.execute("""
        INSERT INTO shapes (uuid, point_uuids, line_uuids, color, movements)
        VALUES (?, ?, ?, ?, ?)
    """, (
        shape_uuid,                         # uuid
        json.dumps([point_uuid]),          # point_uuids
        json.dumps([line_uuid]),           # line_uuids
        json.dumps([1.0, 0.0, 0.0]),      # color (Red)
        json.dumps({})                    # movements
    ))

    # Insert sample music entry (e.g., play A4 (440Hz) for 1000ms at timestamp 0ms using instrument 'sine')
    cur.execute("""
        INSERT INTO music (timestamp_ms, notes, durations, instrument_id)
        VALUES (?, ?, ?, ?)
    """, (
        0,              # timestamp_ms
        json.dumps([440]), # notes (A4)
        json.dumps([1000]), # durations (1000ms)
        'sine'          # instrument_id
    ))

     # Insert sample speech entry (e.g., say "Hello World" at 2000ms using voice 'default')
    cur.execute("""
        INSERT INTO speech (sentence, start_time_ms, voice_id)
        VALUES (?, ?, ?)
    """, (
        "Hello World", # sentence
        2000,          # start_time_ms
        'default'      # voice_id
    ))


    conn.commit()
    conn.close()
    print("Example data seeded.")

def main():
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="ZeroInit - Initialize the ZeroEngine database")
    ap.add_argument("--db", default=DB_FILE, help="Path to SQLite DB (default graphics.db)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing DB file")
    ap.add_argument("--seed-example", action="store_true", help="Insert a tiny example after creating DB")
    args = ap.parse_args()

    if os.path.exists(args.db) and not args.overwrite:
        print(f"DB '{args.db}' already exists. Use --overwrite to recreate.")
        sys.exit(1)

    create_database(args.db, overwrite=args.overwrite)

    if args.seed_example:
        seed_example_data(args.db)

if __name__ == "__main__":
    main()