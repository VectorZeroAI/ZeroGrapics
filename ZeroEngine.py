# ZeroEngine.py - improved rendering for ZeroGraphics with audio support
import sys
import ctypes
import math
import time
import pygame
from pygame.locals import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from ZeroInterpreter import ZeroInterpreter
from ZeroAudio import ZeroAudioPlayer # Import the audio player

# default fallback config values
DEFAULT_CONFIG = {
    "CAMERA_POSITION": (0.0, 0.0, 6.0),
    "POINT_COLOR": (1.0, 0.6, 0.2),
    "LINE_COLOR": (0.9, 0.9, 0.9),
    "LINE_THICKNESS": 2.0,
    "SHAPE_FILL_COLOR": (0.5, 0.5, 1.0),
    "BACKGROUND_COLOR": (0.02, 0.02, 0.03),
    "WINDOW_SIZE": (1280, 720)
}

# FIXED: Added proper Config import with error handling
try:
    from Config import Config
except ImportError:
    Config = None # Will fall back to DEFAULT_CONFIG

def cfg(name):
    if Config is not None and hasattr(Config, name):
        return getattr(Config, name)
    return DEFAULT_CONFIG.get(name, None)

class ZeroEngine:
    def __init__(self, db_path, width=None, height=None, target_fps=60): # Renamed sample_rate to target_fps for clarity
        pygame.init()
        self.width, self.height = width or cfg("WINDOW_SIZE")[0], height or cfg("WINDOW_SIZE")[1]
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("ZeroEngine")

        # --- Fix: Initialize OpenGL settings ---
        glEnable(GL_DEPTH_TEST) # Enable depth testing for 3D
        glDepthFunc(GL_LESS)   # Closer objects obscure farther ones
        glLineWidth(cfg("LINE_THICKNESS")) # Set line thickness

        # --- Initialize VBOs ---
        self.vbo_points = glGenBuffers(1)
        self.vbo_lines = glGenBuffers(1)
        self.vbo_triangles = glGenBuffers(1)

        # --- Fix: Storage for line segment counts (Issue #1) ---
        self.line_segment_counts = []
        self.total_line_vertices = 0 # Track total vertices for line VBO offset

        # --- Initialize Interpreter and Audio ---
        self.interpreter = ZeroInterpreter(db_path)
        self.audio_player = ZeroAudioPlayer(db_path)

        # --- Timing and Control ---
        self.target_fps = target_fps
        self.clock = pygame.time.Clock()
        self.current_tick = 0
        self.running = True # Add a flag to control the main loop

    def prepare_frame_gpu(self, frame_data):
        """Uploads point, line, and triangle data to GPU VBOs."""
        # --- Points ---
        points = frame_data.get('points', [])
        if points:
            point_array = np.array([[p['coordinates'][0], p['coordinates'][1], p['coordinates'][2]] for p in points], dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
            glBufferData(GL_ARRAY_BUFFER, point_array.nbytes, point_array, GL_DYNAMIC_DRAW)
            self.point_count = len(points)
        else:
            self.point_count = 0

        # --- Lines (Fix: Handle segments correctly) ---
        lines = frame_data.get('lines', [])
        self.line_segment_counts = [] # Reset counts for this frame
        line_vertices_list = []
        for line in lines:
            # --- Fix: Calculate Bezier curve points here (Option B) ---
            p0 = np.array(line['endpoints'][0]['coordinates'], dtype=np.float32)
            p1 = np.array(line['endpoints'][1]['coordinates'], dtype=np.float32)
            pc = np.array(line['pull_point']['coordinates'], dtype=np.float32)
            power = line.get('pull_power', 1.0)

            # Calculate number of samples based on distance and power (basic adaptive sampling)
            # Distance influences the number of points for smoother curves on longer lines.
            distance = np.linalg.norm(p1 - p0)
            num_samples = max(2, int(distance * 20 * power)) # Adjust multiplier (20) as needed

            # Sample the quadratic Bezier curve
            # B(t) = (1-t)^2 * P0 + 2*(1-t)*t * Pc + t^2 * P1
            segment_points = []
            for i in range(num_samples + 1): # +1 to include the end point
                t = i / num_samples
                bt = (1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * pc + t * t * p1
                segment_points.append(bt)

            self.line_segment_counts.append(len(segment_points))
            line_vertices_list.extend(segment_points)

        if line_vertices_list:
             # Create a single large array for all line vertices
            line_array = np.array(line_vertices_list, dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
            glBufferData(GL_ARRAY_BUFFER, line_array.nbytes, line_array, GL_DYNAMIC_DRAW)
            self.total_line_vertices = len(line_vertices_list) # Total vertices uploaded
        else:
            self.total_line_vertices = 0

        # --- Shapes (Triangles) ---
        shapes = frame_data.get('shapes', [])
        triangle_vertices = []
        for shape in shapes:
            # Basic triangulation: Fan triangulation assuming points form a convex polygon
            point_uuids = shape.get('point_uuids', [])
            if len(point_uuids) < 3:
                continue

            # Find coordinates for points in this shape
            shape_points = [p for p in points if p['uuid'] in point_uuids]
            if len(shape_points) < 3:
                 continue # Not enough points fetched for this shape

            coords = np.array([p['coordinates'] for p in shape_points], dtype=np.float32)

            # Fan triangulation: Pick first point as center, connect to consecutive pairs
            center = coords[0]
            for i in range(1, len(coords) - 1):
                triangle_vertices.extend([center, coords[i], coords[i+1]])

        if triangle_vertices:
            triangle_array = np.array(triangle_vertices, dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_triangles)
            glBufferData(GL_ARRAY_BUFFER, triangle_array.nbytes, triangle_array, GL_DYNAMIC_DRAW)
            self.triangle_count = len(triangle_vertices) # Number of vertices (multiples of 3)
        else:
            self.triangle_count = 0

    def render_frame(self, frame_data):
        """Renders the current frame using OpenGL."""
        # Clear screen
        glClearColor(*cfg("BACKGROUND_COLOR"), 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up projection and view (Simple setup)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect_ratio = self.width / self.height if self.height != 0 else 1.0
        gluPerspective(45, aspect_ratio, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera_pos = cfg("CAMERA_POSITION")
        gluLookAt(*camera_pos, 0, 0, 0, 0, 1, 0)

        # Enable vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)

        # --- Render Triangles (Shapes) ---
        if self.triangle_count > 0:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_triangles)
            glVertexPointer(3, GL_FLOAT, 0, None)
            color = cfg("SHAPE_FILL_COLOR")
            glColor3f(*color)
            glDrawArrays(GL_TRIANGLES, 0, self.triangle_count) # Draw all triangles

        # --- Render Lines ---
        if self.total_line_vertices > 0 and self.line_segment_counts:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
            glVertexPointer(3, GL_FLOAT, 0, None)
            color = cfg("LINE_COLOR")
            glColor3f(*color)

            # --- Fix: Draw each line segment correctly (Issue #1) ---
            start_index = 0
            for count in self.line_segment_counts:
                if count >= 2: # Need at least 2 points for a line strip
                    glDrawArrays(GL_LINE_STRIP, start_index, count)
                start_index += count

        # --- Render Points ---
        if self.point_count > 0:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
            glVertexPointer(3, GL_FLOAT, 0, None)
            color = cfg("POINT_COLOR")
            glColor3f(*color)
            glDrawArrays(GL_POINTS, 0, self.point_count)

        # Disable vertex arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

        # Update display
        pygame.display.flip()

    def handle_events(self):
        """Handles Pygame events like window resize and quit."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == VIDEORESIZE:
                self.width, self.height = event.size
                pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height) # Update OpenGL viewport
        return True

    # FIXED: Corrected indentation for the run method docstring (Issue #5)
    def run(self):
        """Main rendering loop."""
        # Initialize running flag
        self.running = True
        try:
            while self.running:
                # Update audio for the current time
                self.audio_player.update(self.current_tick)

                # Read frame from interpreter
                frame = self.interpreter.read_full(self.current_tick)

                # Convert and upload to GPU
                self.prepare_frame_gpu(frame)

                # Render
                self.render_frame(frame)

                # Debug caption
                pygame.display.set_caption(f"ZeroEngine| Time: {self.current_tick}ms| Points: {len(frame.get('points',[]))}| Lines: {len(frame.get('lines',[]))}| Shapes: {len(frame.get('shapes',[]))}")

                # Maintain frame rate
                self.clock.tick(self.target_fps)

                # Handle events and update running flag
                self.running = self.handle_events()

                # Advance time
                self.current_tick += 1 # Increment time by 1ms per frame

        except Exception as e:
            print(f"An error occurred during the run loop: {e}")
        finally:
            # Cleanup
            self.audio_player.cleanup()
            self.cleanup() # Call cleanup for VBOs
            pygame.quit()

    # FIXED: Added cleanup method for VBOs (Issue #4)
    def cleanup(self):
        """Clean up OpenGL resources."""
        try:
            glDeleteBuffers(1, [self.vbo_points])
            glDeleteBuffers(1, [self.vbo_lines])
            glDeleteBuffers(1, [self.vbo_triangles])
            print("VBOs deleted successfully.")
        except Exception as e:
            print(f"Error deleting VBOs: {e}")
        # Note: Pygame.quit is called in run(), and audio cleanup in audio_player.cleanup()

# --- Main execution block (moved outside the class) ---
def main():
    import argparse
    import os
    ap = argparse.ArgumentParser(description="ZeroEngine - Graphics and Audio Renderer")
    ap.add_argument("--db", default="graphics.db", help="Path to SQLite DB (default graphics.db)")
    ap.add_argument("--width", type=int, help="Window width")
    ap.add_argument("--height", type=int, help="Window height")
    # Assuming target_fps might be configurable too, though not in original snippet
    ap.add_argument("--fps", type=int, default=60, help="Target FPS (default 60)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print("DB not found:", args.db)
        print("Run ZeroInit.py to create the DB first.")
        sys.exit(1)

    engine = ZeroEngine(args.db, args.width, args.height, target_fps=args.fps)
    engine.run()

if __name__ == "__main__":
    main()
