# ZeroEngine.py - Kivy-based renderer for ZeroGraphics with audio support
import kivy
kivy.require('2.1.0')
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Point, Line, Triangle, Color, InstructionGroup, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.config import Config as KivyConfig
import math
import sqlite3
import json
import time
import sys
from ZeroInterpreter import ZeroInterpreter
from ZeroAudio import ZeroAudioPlayer  # Import the audio player

# Default config values (used if Config module is not available)
DEFAULT_CONFIG = {
    "CAMERA_POSITION": (0.0, 0.0, 6.0),
    "POINT_COLOR": (1.0, 0.6, 0.2, 1.0),  # RGBA
    "LINE_COLOR": (0.9, 0.9, 0.9, 1.0),    # RGBA
    "SHAPE_FILL_COLOR": (0.5, 0.5, 1.0, 0.8),  # RGBA with transparency
    "BACKGROUND_COLOR": (0.02, 0.02, 0.03, 1.0),  # RGBA
    "LINE_THICKNESS": 2.0,
    "POINT_SIZE": 5.0,
    "FPS": 60,
    "WINDOW_SIZE": (1280, 720)
}

# Try to import custom Config, fall back to defaults if not available
try:
    from Config import Config as CustomConfig
except ImportError:
    CustomConfig = None

def cfg(name):
    """Get configuration value, trying CustomConfig first, then DEFAULT_CONFIG"""
    if CustomConfig and hasattr(CustomConfig, name):
        return getattr(CustomConfig, name)
    return DEFAULT_CONFIG.get(name, None)

class Graphics:
    """Core graphics engine that manages the animation timeline and rendering logic"""

    def __init__(self, db_path):
        """Initialize the graphics engine with database path"""
        self.db_path = db_path
        self.pause = 1.0 / cfg("FPS")
        self.model_2d = None
        self.model_3d = None
        self.frames = []
        self.current_frame_index = 0
        self.interpreter = ZeroInterpreter(db_path)
        self.audio_player = ZeroAudioPlayer(db_path)
        self.current_tick = 0
        self.widget = None  # Will be set by the Kivy app

    def calculate_total_frames(self, fps: int):
        """
        Calculate the amount of frames the animation should have,
        according to FPS and the maximal ms defined by the DB.
        Store results in self.frames.
        """
        # Query the DB for the maximum timestamp across all elements
        max_time = 0

        # Check points for movements
        for uuid, point in self.interpreter.points.items():
            if point["movements"]:
                last_movement = point["movements"][-1]
                end_time = last_movement[0] + last_movement[1]
                max_time = max(max_time, end_time)

        # Check lines for movements
        for uuid, line in self.interpreter.lines.items():
            if line["movements"]:
                last_movement = line["movements"][-1]
                end_time = last_movement[0] + last_movement[1]
                max_time = max(max_time, end_time)

        # Check shapes for movements
        for uuid, shape in self.interpreter.shapes.items():
            if shape["movements"]:
                last_movement = shape["movements"][-1]
                end_time = last_movement[0] + last_movement[1]
                max_time = max(max_time, end_time)

        # Check music events
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Check music
        cur.execute("SELECT MAX(timestamp_ms) FROM music")
        music_max = cur.fetchone()[0] or 0
        max_time = max(max_time, music_max)

        # Check speech
        cur.execute("SELECT MAX(start_time_ms) FROM speech")
        speech_max = cur.fetchone()[0] or 0
        max_time = max(max_time, speech_max)

        conn.close()

        # Add buffer (1 second)
        max_time += 1000

        # Calculate frames
        if max_time <= 0:
            max_time = 10000  # Default to 10 seconds if no data

        # Create frame timestamps (in ms)
        self.frames = [i for i in range(0, max_time, 1000 // fps)]



    def project_point(self, point_3d):
        """Project 3D point to 2D using camera settings from Config (fixed math)."""
        xw, yw, zw = point_3d

    # Camera position from config (cx, cy, cz)
        camera_pos = cfg("CAMERA_POSITION")
        cx, cy, cz = camera_pos

    # Transform to camera space.
    # We assume camera looks toward negative world-z and is located at cz.
    # So points in front of camera should have positive z_cam = cz - zw
        x_cam = xw - cx
        y_cam = yw - cy
        z_cam = cz - zw

    # Field of view (vertical)
        fov = 45.0  # degrees (same as before)
        fov_rad = math.radians(fov)
        scale = 1.0 / math.tan(fov_rad / 2.0)

    # Aspect ratio (width / height)
        aspect = (Window.width / Window.height) if Window.height != 0 else 1.0

    # Avoid extremely small or negative z that would blow up projection
        min_z = 0.01
        if z_cam < min_z:
            z_cam = min_z

    # Standard perspective projection:
    # x_ndc = (x_cam * scale) / (z_cam * aspect)
    # y_ndc = (y_cam * scale) / z_cam
        x_ndc = (x_cam * scale) / (z_cam * aspect)
        y_ndc = (y_cam * scale) / z_cam

    # Map normalized device coords (-1..1) to screen coordinates
        x_screen = (x_ndc + 1.0) * 0.5 * Window.width
        y_screen = (1.0 - (y_ndc + 1.0) * 0.5) * Window.height
    # Note: if your Y axes appear flipped, try using:
    # y_screen = (y_ndc + 1.0) * 0.5 * Window.height

        return (x_screen, y_screen)




    def transform_3d_to_2d(self, model_3d):
        """
        Transform the 3D model into 2D coordinates that can be rendered.
        """
        model_2d = {
            "points": {},
            "lines": {},
            "shapes": {}
        }

        # Transform points
        for uuid, point in model_3d["points"].items():
            model_2d["points"][uuid] = self.project_point(point)

        # Transform lines
        for uuid, line in model_3d["lines"].items():
            # Transform the sampled points
            sampled_points_2d = [self.project_point(p) for p in line["sampled_points"]]
            model_2d["lines"][uuid] = {
                "sampled_points": sampled_points_2d
            }






        # --- Replace the current "Transform shapes" block in transform_3d_to_2d with this ---

        # Helper utilities
        def pts_equal_3d(a, b, eps=1e-6):
            return abs(a[0]-b[0]) < eps and abs(a[1]-b[1]) < eps and abs(a[2]-b[2]) < eps

        def remove_adjacent_duplicates(points2d, eps=1e-3):
            if not points2d:
                return points2d
            out = [points2d[0]]
            for p in points2d[1:]:
                last = out[-1]
                if (abs(p[0]-last[0]) > eps) or (abs(p[1]-last[1]) > eps):
                    out.append(p)
            # make sure first != last
            if len(out) > 1:
                if abs(out[0][0]-out[-1][0]) < eps and abs(out[0][1]-out[-1][1]) < eps:
                    out.pop()
            return out

        # Transform shapes
        for uuid, shape in model_3d["shapes"].items():
            boundary_2d = []

            # explicit line ids preferred
            line_ids = shape.get("line_uuids") or []

            # prepare the list of 3D corner points, in order
            corner_points_3d = []
            for p_uuid in shape.get("point_uuids", []):
                if p_uuid in model_3d["points"]:
                    corner_points_3d.append(model_3d["points"][p_uuid])

            # Helper to append sampled points for a line while ensuring orientation
            def append_line_samples_for_edge(a3, b3, line_rec):
                sampled = line_rec.get("sampled_points", [])
                if not sampled:
                    # fallback to endpoints
                    boundary_2d.append(self.project_point(a3))
                    return
                # Determine which end of sampled corresponds to a3 or b3
                if pts_equal_3d(sampled[0], a3):
                    seq = sampled
                elif pts_equal_3d(sampled[-1], a3):
                    seq = list(reversed(sampled))
                else:
                    # If neither sampled endpoint matches exactly, try matching to b3
                    if pts_equal_3d(sampled[0], b3):
                        seq = list(reversed(sampled))
                    elif pts_equal_3d(sampled[-1], b3):
                        seq = sampled
                    else:
                        # Unknown orientation — still use sampled as-is
                        seq = sampled
                for p in seq:
                    boundary_2d.append(self.project_point(p))

            if line_ids:
                # Use explicit line list; assume lines are already given in correct order for the shape
                for lid in line_ids:
                    line_rec = model_3d["lines"].get(lid)
                    if not line_rec:
                        continue
                    # NOTE: line_rec['endpoints'] in model_3d may be uuids or positions depending on interpreter.
                    # We prefer matching using coordinates if available.
                    # Try to fetch endpoints as coords:
                    ep = line_rec.get("endpoints", [])
                    a3 = None
                    b3 = None
                    if ep:
                        # If endpoints are uuids, resolve
                        if isinstance(ep[0], str) and ep[0] in model_3d["points"]:
                            a3 = model_3d["points"][ep[0]]
                        elif isinstance(ep[0], (list, tuple)):
                            a3 = ep[0]
                        if len(ep) > 1:
                            if isinstance(ep[1], str) and ep[1] in model_3d["points"]:
                                b3 = model_3d["points"][ep[1]]
                            elif isinstance(ep[1], (list, tuple)):
                                b3 = ep[1]
                    # fallback: try to pick from shape corner order
                    if a3 is None or b3 is None:
                        # try to find a line whose endpoints match two consecutive corners
                        pass
                    # Append sampled points oriented from a3->b3 if possible
                    if a3 is not None and b3 is not None:
                        append_line_samples_for_edge(a3, b3, line_rec)
                    else:
                        # If we couldn't determine endpoints, just append sampled (best-effort)
                        for p in line_rec.get("sampled_points", []):
                            boundary_2d.append(self.project_point(p))

            else:
                # No explicit lines: auto-detect lines for consecutive corners
                n = len(corner_points_3d)
                if n >= 2:
                    for i in range(n):
                        a3 = corner_points_3d[i]
                        b3 = corner_points_3d[(i+1) % n]
                        # search for a line between a3 and b3
                        found_line = None
                        for lid, line_rec in model_3d["lines"].items():
                            ep = line_rec.get("endpoints", [])
                            if not ep or len(ep) < 2:
                                continue
                            # resolve endpoints to coords if needed
                            ep0 = ep[0]
                            ep1 = ep[1]
                            if isinstance(ep0, str) and ep0 in model_3d["points"]:
                                ep0 = model_3d["points"][ep0]
                            if isinstance(ep1, str) and ep1 in model_3d["points"]:
                                ep1 = model_3d["points"][ep1]
                            if pts_equal_3d(ep0, a3) and pts_equal_3d(ep1, b3) or \
                               pts_equal_3d(ep0, b3) and pts_equal_3d(ep1, a3):
                                found_line = line_rec
                                break
                        if found_line:
                            append_line_samples_for_edge(a3, b3, found_line)
                        else:
                            # fallback: straight edge endpoints
                            boundary_2d.append(self.project_point(a3))

            # Cleanup duplicates and very-close points
            boundary_2d = remove_adjacent_duplicates(boundary_2d)

            # Final fallback: if empty, use projected corner points
            if not boundary_2d:
                for p3 in corner_points_3d:
                    boundary_2d.append(self.project_point(p3))

            model_2d["shapes"][uuid] = {
                "points_2d": boundary_2d,
                "color": shape.get("color", (1.0, 1.0, 1.0))
            }


        return model_2d




    def render(self, model_2d):
        """
        Render the 2D model using Kivy graphics.
        Check what the DB provides, and what it doesn't provide. Everything that the DB doesn't provide is config defined.
        """
        if not self.widget:
            return

        # Clear previous drawings
        self.widget.canvas.clear()

        # Set background color
        with self.widget.canvas:
            Color(*cfg("BACKGROUND_COLOR"))
            Rectangle(pos=(0, 0), size=(Window.width, Window.height))

        with self.widget.canvas:
            # Draw points
            point_coords = []
            for uuid, point in model_2d["points"].items():
                x, y = point
                point_coords.extend([x, y])

            if point_coords:
                Color(*cfg("POINT_COLOR"))
                Point(points=point_coords, pointsize=cfg("POINT_SIZE"))

            # Draw lines
            for uuid, line in model_2d["lines"].items():
                line_coords = []
                for point in line["sampled_points"]:
                    x, y = point
                    line_coords.extend([x, y])

                if line_coords:
                    Color(*cfg("LINE_COLOR"))
                    Line(points=line_coords, width=cfg("LINE_THICKNESS"))

            # Draw shapes (filled)
            for uuid, shape in model_2d["shapes"].items():
                points = shape["points_2d"]
                if len(points) >= 3:
                    # Set shape color with transparency
                    color = shape["color"]
                    # Ensure we have 4 components (RGBA)
                    if len(color) == 3:
                        color = (color[0], color[1], color[2], 0.8)
                    Color(*color)

                    # Simple fan triangulation for convex polygons
                    for i in range(1, len(points) - 1):
                        Triangle(points=[
                            points[0][0], points[0][1],
                            points[i][0], points[i][1],
                            points[i+1][0], points[i+1][1]
                        ])

    def frame_full(self, ms: int):
        """Process a single frame at the specified millisecond timestamp"""
        self.model_3d = self.interpreter.read_full(ms)
        self.model_2d = self.transform_3d_to_2d(self.model_3d)
        self.render(self.model_2d)
        # Update audio for this timestamp
        self.audio_player.update(ms)
        # Update debug info
        self.current_tick = ms

    def play(self, dt=None):
        """
        Do not change this loop. This loop should act as the anchor that the program you create works with.
        """
        if not self.frames:
            self.calculate_total_frames(cfg("FPS"))

        if self.current_frame_index >= len(self.frames):
            self.current_frame_index = 0  # Loop back to start

        ms = self.frames[self.current_frame_index]
        self.frame_full(ms)

        # Update debug label if available
        if hasattr(self.widget, 'update_debug_info'):
            self.widget.update_debug_info(ms, 
                len(self.model_3d.get('points', [])),
                len(self.model_3d.get('lines', [])),
                len(self.model_3d.get('shapes', [])))

        self.current_frame_index += 1

class GraphicsWidget(Widget):
    """Kivy widget that handles the actual rendering canvas"""

    def __init__(self, graphics, **kwargs):
        super().__init__(**kwargs)
        self.graphics = graphics
        self.graphics.widget = self  # Give Graphics access to the widget's canvas
        self.graphics.calculate_total_frames(cfg("FPS"))

        # Create debug label
        from kivy.uix.label import Label
        self.debug_label = Label(
            text="Initializing...",
            pos=(10, Window.height - 30),
            size=(200, 30),
            halign='left',
            valign='middle',
            font_size=14
        )
        self.add_widget(self.debug_label)

    def update_debug_info(self, ms, num_points, num_lines, num_shapes):
        """Update the debug information label"""
        self.debug_label.text = f"Time: {ms}ms | Points: {num_points} | Lines: {num_lines} | Shapes: {num_shapes}"

class ZeroEngineApp(App):
    """Main Kivy application class for the ZeroEngine"""

    def __init__(self, db_path, **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
        self.graphics = None

    def build(self):
        """Build the application UI"""
        # Set window properties
        window_size = cfg("WINDOW_SIZE")
        Window.size = (window_size[0], window_size[1])
        Window.clearcolor = cfg("BACKGROUND_COLOR")

        # Create the graphics engine and widget
        self.graphics = Graphics(self.db_path)
        return GraphicsWidget(self.graphics)

def main():
    """Main entry point for the application"""
    import argparse
    import os

    ap = argparse.ArgumentParser(description="ZeroEngine - Kivy-based Graphics and Audio Renderer")
    ap.add_argument("--db", default="graphics.db", help="Path to SQLite DB (default graphics.db)")
    ap.add_argument("--width", type=int, help="Window width")
    ap.add_argument("--height", type=int, help="Window height")
    ap.add_argument("--fps", type=int, default=cfg("FPS"), help=f"Target FPS (default {cfg('FPS')})")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print("DB not found:", args.db)
        print("Run ZeroInit.py to create the DB first.")
        sys.exit(1)

    # Set window size if provided
    if args.width and args.height:
        KivyConfig.set('graphics', 'width', str(args.width))
        KivyConfig.set('graphics', 'height', str(args.height))

    # Update config with command line FPS
    if not hasattr(CustomConfig, 'FPS') and not CustomConfig is None:
        CustomConfig.FPS = args.fps

    print(f"Starting ZeroEngine with {args.fps} FPS")
    app = ZeroEngineApp(args.db)
    app.run()

if __name__ == "__main__":
    main()