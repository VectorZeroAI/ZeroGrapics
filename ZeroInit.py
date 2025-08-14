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
        print(f"Existing database '{db_path}' removed.")
    # Connect to DB
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Enable foreign keys (future-proofing)
    c.execute("PRAGMA foreign_keys = ON;")
    # ---------------------------
    # Table 1: Points
    # ---------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS points (
    coordinates TEXT NOT NULL, -- "{x;y;z}"
    uuid TEXT PRIMARY KEY, -- unique point identifier
    connected_points TEXT, -- "{uuid1;uuid2;...}"
    movements TEXT -- JSON string: "{(x;y;[a;b;c]);...}"
    );
    """)
    # ---------------------------
    # Table 2: Lines
    # ---------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS lines (
    uuid TEXT PRIMARY KEY, -- unique line identifier
    endpoints TEXT NOT NULL, -- "{point_uuid1;point_uuid2}"
    pull_point TEXT NOT NULL, -- "{x;y;z}"
    pull_power REAL NOT NULL, -- numeric value
    movements TEXT -- JSON string: "{(x;y;[a;b;c;d]);...}"
    );
    """)
    # ---------------------------
    # Table 3: Shapes (FIXED: Added uuid column)
    # ---------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS shapes (
    uuid TEXT PRIMARY KEY, -- unique shape identifier (FIXED: Added this column)
    point_uuids TEXT NOT NULL, -- "{point_uuid1;point_uuid2;...}"
    line_uuids TEXT NOT NULL, -- "{line_uuid1;line_uuid2;...}"
    color TEXT NOT NULL, -- "{r;g;b}"
    movements TEXT -- JSON string: "{(x;y;[a;b;c]);...}"
    );
    """)
    conn.commit()
    conn.close()
    print(f"Database '{db_path}' created/verified with required tables.")

def seed_example_data(db_path=DB_FILE):
    """Optional: Inserts one sample row per table for debugging."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Example point
    point_uuid = str(uuid.uuid4())
    c.execute("""
    INSERT INTO points (coordinates, uuid, connected_points, movements)
    VALUES (?, ?, ?, ?)
    """, ("{0;0;0}", point_uuid, "{}", "{}"))
    # Example line
    line_uuid = str(uuid.uuid4())
    c.execute("""
    INSERT INTO lines (uuid, endpoints, pull_point, pull_power, movements)
    VALUES (?, ?, ?, ?, ?)
    """, (line_uuid, f"{{{point_uuid};{point_uuid}}}", "{0;0;0}", 1.0, "{}"))
    # Example shape (FIXED: Added uuid parameter)
    shape_uuid = str(uuid.uuid4())
    c.execute("""
    INSERT INTO shapes (uuid, point_uuids, line_uuids, color, movements)
    VALUES (?, ?, ?, ?, ?)
    """, (shape_uuid, f"{{{point_uuid}}}", f"{{{line_uuid}}}", "{1;0;0}", "{}"))
    conn.commit()
    conn.close()
    print("Example data inserted.")

if __name__ == "__main__":
    create_database(overwrite=False) # Change to True to reset DB
    # seed_example_data() # Uncomment for debug data