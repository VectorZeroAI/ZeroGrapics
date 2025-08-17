# ZeroApp.py - Main Kivy application with embedded ZeroEngine renderer
import os
import json
import sqlite3
import sys

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.clock import Clock

from ZeroFiller import process_json_input, process_legacy_text, clear_tables

# Import engine pieces (these are defined in your existing ZeroEngine.py)
# ZeroEngine must not execute app.run() on import (it doesn't if guarded by __main__).
from ZeroEngine import Graphics, GraphicsWidget, cfg

DB_PATH = "graphics.db"


class ZeroApp(App):
    def build(self):
        Window.size = (1000, 700)
        Window.title = "ZeroGraphics Controller"

        # Root layout: horizontal split — left controls, right engine viewport
        self.root_layout = BoxLayout(orientation="horizontal", spacing=10, padding=10)

        # Left column: input + buttons
        left_col = BoxLayout(orientation="vertical", size_hint=(0.45, 1), spacing=10)

        left_col.add_widget(Label(text="Paste LLM JSON Here:", size_hint=(1, 0.04)))

        self.json_input = TextInput(
            multiline=True,
            size_hint=(1, 0.76),
            font_size=14,
            background_color=(0.95, 0.95, 0.98, 1),
            foreground_color=(0.05, 0.05, 0.05, 1)
        )
        left_col.add_widget(self.json_input)

        # Buttons row
        btn_layout = BoxLayout(size_hint=(1, 0.12), spacing=8)

        self.apply_btn = Button(text="Apply", on_press=self.apply_instructions)
        btn_layout.add_widget(self.apply_btn)

        self.play_btn = Button(text="Play", on_press=self.toggle_play, background_color=(0.2, 0.4, 0.8, 1))
        btn_layout.add_widget(self.play_btn)

        self.stop_btn = Button(text="Stop", on_press=self.stop_animation, disabled=True)
        btn_layout.add_widget(self.stop_btn)

        left_col.add_widget(btn_layout)

        # Status label
        self.status_label = Label(text="Ready", size_hint=(1, 0.08), color=(0.2, 0.2, 0.2, 1))
        left_col.add_widget(self.status_label)

        # Right column: engine viewport placeholder
        self.right_col = BoxLayout(orientation="vertical", size_hint=(0.55, 1))
        self.viewport_label = Label(text="Viewport (press Play to load)", size_hint=(1, 0.02))
        self.right_col.add_widget(self.viewport_label)

        # Add columns to root
        self.root_layout.add_widget(left_col)
        self.root_layout.add_widget(self.right_col)

        # Engine-related state
        self.graphics = None
        self.engine_widget = None
        self._scheduled_event = None
        self.engine_running = False

        # Ensure DB exists
        if not os.path.exists(DB_PATH):
            self.initialize_database()

        return self.root_layout

    def initialize_database(self):
        """Create database if it doesn't exist"""
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # Create tables (same as before)
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

    def show_error(self, message):
        popup = Popup(title="Error", content=Label(text=message), size_hint=(0.7, 0.3))
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 2.0)
        self.status_label.text = "Error: " + message

    def apply_instructions(self, instance):
        """Process JSON input and update database — ensure commit/close before running engine"""
        json_text = self.json_input.text.strip()
        if not json_text:
            self.show_error("Please enter JSON instructions")
            return

        try:
            conn = sqlite3.connect(DB_PATH)

            # Optional: set WAL to reduce lock contention if engine reads while writing
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except Exception:
                pass

            clear_tables(conn)

            try:
                data = json.loads(json_text)
                inserted = process_json_input(conn, data)
            except json.JSONDecodeError:
                inserted = process_legacy_text(conn, json_text)

            conn.commit()
            conn.close()

            summary = (
                f"Applied successfully!\n"
                f"Points: {inserted.get('points',0)} | Lines: {inserted.get('lines',0)}\n"
                f"Shapes: {inserted.get('shapes',0)} | Music: {inserted.get('music',0)}\n"
                f"Speech: {inserted.get('speech',0)}"
            )
            self.status_label.text = summary

        except Exception as e:
            self.show_error(f"Error applying instructions: {str(e)}")

    def _create_engine_if_needed(self):
        """Create Graphics + GraphicsWidget if not already present."""
        if self.graphics is None:
            self.graphics = Graphics(DB_PATH)
        if self.engine_widget is None:
            # Use a fixed size_hint so it fills the right column
            self.engine_widget = GraphicsWidget(self.graphics, size_hint=(1, 1))
            # Add the widget to the right column (replace placeholder)
            self.right_col.clear_widgets()
            self.right_col.add_widget(self.engine_widget)

            # Give Graphics a reference to widget (GraphicsWidget's __init__ does this)
            self.graphics.widget = self.engine_widget

            # Ensure frames are calculated now that interpreter is ready
            try:
                self.graphics.calculate_total_frames(cfg("FPS"))
            except Exception:
                # If cfg or other pieces aren't available, don't crash; leave default behaviour
                pass

    def toggle_play(self, instance):
        """Start or pause the animation loop (single process)"""
        if self.engine_running:
            self.stop_animation(instance)
        else:
            self.start_animation()

    def start_animation(self):
        """Start the embedded graphics loop."""
        if not os.path.exists(DB_PATH):
            self.show_error("Database not found. Apply instructions first.")
            return

        # Guarantee DB is flushed/closed by previous apply
        # (apply_instructions commits+closes, but we make sure)
        try:
            c = sqlite3.connect(DB_PATH)
            c.execute("PRAGMA journal_mode=WAL;")
            c.commit()
            c.close()
        except Exception:
            pass

        # Make engine and widget if needed, and schedule the play loop
        self._create_engine_if_needed()

        if self._scheduled_event:
            # already scheduled (shouldn't normally happen)
            self.show_error("Engine already scheduled.")
            return

        fps = cfg("FPS") or 60
        interval = 1.0 / float(fps)

        # Schedule the graphics.play on the Kivy Clock — runs in the same thread/loop
        self._scheduled_event = Clock.schedule_interval(self.graphics.play, interval)
        self.engine_running = True
        self.play_btn.text = "Pause"
        self.stop_btn.disabled = False
        self.status_label.text = "Animation running..."

    def stop_animation(self, instance=None):
        """Stop the scheduled loop and optionally remove the widget."""
        if self._scheduled_event:
            try:
                self._scheduled_event.cancel()
            except Exception:
                pass
            self._scheduled_event = None

        self.engine_running = False
        self.play_btn.text = "Play"
        self.stop_btn.disabled = True
        self.status_label.text = "Animation stopped."

        # Optionally keep the widget visible so user can inspect; if you prefer to remove it replace below:
        # self._destroy_engine_widget()

    def _destroy_engine_widget(self):
        """Remove and dereference engine widget — useful if you want to fully unload engine."""
        if self.engine_widget:
            try:
                self.right_col.remove_widget(self.engine_widget)
            except Exception:
                pass
            self.engine_widget = None
        self.graphics = None

if __name__ == "__main__":
    ZeroApp().run()