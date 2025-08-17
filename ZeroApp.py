# ZeroApp.py - Main Kivy application with GUI for ZeroGraphics system
import os
import json
import subprocess
import sqlite3
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.clock import Clock
from ZeroFiller import process_json_input, process_legacy_text, clear_tables

DB_PATH = "graphics.db"

class ZeroApp(App):
    def build(self):
        # Create main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # JSON input field
        self.json_input = TextInput(
            multiline=True,
            size_hint=(1, 0.7),
            font_size=14,
            background_color=(0.9, 0.9, 0.95, 1),
            foreground_color=(0.1, 0.1, 0.1, 1)
        )
        layout.add_widget(Label(text="Paste LLM JSON Here:", size_hint=(1, 0.05)))
        layout.add_widget(self.json_input)
        
        # Button panel
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=10)
        
        # Apply button
        apply_btn = Button(
            text="Apply",
            background_color=(0.2, 0.6, 0.2, 1),
            on_press=self.apply_instructions
        )
        btn_layout.add_widget(apply_btn)
        
        # Play button
        play_btn = Button(
            text="Play",
            background_color=(0.2, 0.4, 0.8, 1),
            on_press=self.play_animation
        )
        btn_layout.add_widget(play_btn)
        
        layout.add_widget(btn_layout)
        
        # Status label
        self.status_label = Label(
            text="Ready",
            size_hint=(1, 0.1),
            color=(0.3, 0.3, 0.3, 1)
        )
        layout.add_widget(self.status_label)
        
        # Initialize database if needed
        if not os.path.exists(DB_PATH):
            self.initialize_database()
            
        return layout

    def initialize_database(self):
        """Create database if it doesn't exist"""
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Create tables (simplified version of ZeroInit)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS points (
                coordinates TEXT NOT NULL,
                uuid TEXT PRIMARY KEY,
                connected_points TEXT,
                movements TEXT
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lines (
                uuid TEXT PRIMARY KEY,
                endpoints TEXT NOT NULL,
                pull_point TEXT,
                pull_power REAL,
                movements TEXT
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shapes (
                uuid TEXT PRIMARY KEY,
                point_uuids TEXT,
                line_uuids TEXT,
                color TEXT,
                movements TEXT
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS music (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                notes TEXT NOT NULL,
                durations TEXT NOT NULL,
                instrument_id TEXT NOT NULL
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS speech (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence TEXT NOT NULL,
                start_time_ms INTEGER NOT NULL,
                voice_id TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        self.status_label.text = "Database initialized"

    def apply_instructions(self, instance):
        """Process JSON input and update database"""
        json_text = self.json_input.text.strip()
        
        if not json_text:
            self.show_error("Please enter JSON instructions")
            return
            
        try:
            conn = sqlite3.connect(DB_PATH)
            
            # Clear existing tables
            clear_tables(conn)
            
            # Try to parse as JSON
            try:
                data = json.loads(json_text)
                inserted = process_json_input(conn, data)
            except json.JSONDecodeError:
                # Fallback to legacy text format
                inserted = process_legacy_text(conn, json_text)
                
            conn.close()
            
            summary = (
                f"Applied successfully!\n"
                f"Points: {inserted['points']} | Lines: {inserted['lines']}\n"
                f"Shapes: {inserted['shapes']} | Music: {inserted['music']}\n"
                f"Speech: {inserted['speech']}"
            )
            self.status_label.text = summary
            
        except Exception as e:
            self.show_error(f"Error applying instructions: {str(e)}")

    def play_animation(self, instance):
        """Launch the animation player"""
        if not os.path.exists(DB_PATH):
            self.show_error("Database not found. Apply instructions first.")
            return
            
        try:
            # Launch ZeroEngine in a separate process
            subprocess.Popen(["python", "ZeroEngine.py", "--db", DB_PATH])
            self.status_label.text = "Animation started..."
        except Exception as e:
            self.show_error(f"Error starting animation: {str(e)}")

    def show_error(self, message):
        """Display error message in a popup"""
        popup = Popup(
            title='Error',
            content=Label(text=message),
            size_hint=(0.7, 0.3)
        )
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 2.0)
        self.status_label.text = "Error: " + message

if __name__ == "__main__":
    # Set window size
    Window.size = (800, 600)
    Window.title = "ZeroGraphics Controller"
    ZeroApp().run()