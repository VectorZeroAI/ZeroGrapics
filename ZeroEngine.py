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
from ZeroAudio import ZeroAudioPlayer  # Import the audio player
import Config

# default fallback config values
DEFAULT_CONFIG = {
    "CAMERA_POSITION": (0.0, 0.0, 6.0),
    "POINT_COLOR": (1.0, 0.6, 0.2),
    "LINE_COLOR": (0.9, 0.9, 0.9),
    "LINE_THICKNESS": 2.0,
    "BACKGROUND_COLOR": (0.02, 0.02, 0.03),
    "WINDOW_SIZE": (1280, 720)
}

def cfg(name):
    return getattr(Config, name) if hasattr(Config, name) else DEFAULT_CONFIG[name]

class ZeroEngine:
    def __init__(self, db_path, width=None, height=None, sample_rate=60):
        pygame.init()
        self.width, self.height = width or cfg("WINDOW_SIZE")[0], height or cfg("WINDOW_SIZE")[1]
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("ZeroEngine")
        self.clock = pygame.time.Clock()
        self.target_fps = sample_rate
        # interpreter
        self.interpreter = ZeroInterpreter(db_path)
        self.current_tick = 0  # ms
        self.paused = False
        self.last_time = pygame.time.get_ticks()
        
        # Initialize audio player
        self.audio_player = ZeroAudioPlayer(db_path)
        
        # GL setup
        self.setup_opengl()
        # GPU buffers (created once, reused)
        self.vbo_points = glGenBuffers(1)
        self.vbo_lines = glGenBuffers(1)
        self.vbo_shapes = glGenBuffers(1)
        # bookkeeping for drawing lines as segments
        self.line_segments = []  # list of (offset, count) per frame
        self.point_count = 0
        self.shape_tri_count = 0
        # a minimal VAO-like state using client arrays (works on compatibility contexts),
        # but you can replace with proper VAO + shaders later.
        glEnableClientState(GL_VERTEX_ARRAY)
    
    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*cfg("BACKGROUND_COLOR"), 1.0)
        # perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, (self.width / float(self.height)), 0.01, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glPointSize(6.0)  # point rendering size (tweakable); spheres later if desired
    
    # --------------------
    # Camera utilities (simplified)
    # --------------------
    def camera_position(self):
        # Fixed camera position
        return np.array(cfg("CAMERA_POSITION"), dtype=np.float32)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == VIDEORESIZE:
                self.width, self.height = event.w, event.h
                pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45.0, (self.width / float(self.height)), 0.01, 1000.0)
                glMatrixMode(GL_MODELVIEW)
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.paused = not self.paused
                    if self.paused:
                        self.audio_player.pause()
                    else:
                        self.audio_player.play()
                elif event.key == K_ESCAPE:
                    return False
                elif event.key == K_m:
                    # Toggle mute
                    if self.audio_player.is_playing:
                        self.audio_player.pause()
                    else:
                        self.audio_player.play()
        return True
    
    # --------------------
    # Upload geometry to GPU
    # --------------------
    def upload_points(self, points_np: np.ndarray):
        # points_np: (N,3) float32
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
        # Dynamic draw because we'll update frequently
        glBufferData(GL_ARRAY_BUFFER, points_np.nbytes, points_np, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.point_count = len(points_np)
    
    def upload_lines(self, lines_np: np.ndarray, segments: list):
        # lines_np: (M,3) float32 concatenated samples of all lines
        # segments: list of (offset, count) in vertex units
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        if len(lines_np) > 0:
            glBufferData(GL_ARRAY_BUFFER, lines_np.nbytes, lines_np, GL_DYNAMIC_DRAW)
            self.line_segments = segments
        else:
            # Skip buffer update when there are no lines
            self.line_segments = []
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def upload_shapes(self, tris_np: np.ndarray):
        # tris_np: (T,3) float32 flattened triangles (T vertices)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_shapes)
        if len(tris_np) > 0:
            glBufferData(GL_ARRAY_BUFFER, tris_np.nbytes, tris_np, GL_DYNAMIC_DRAW)
            self.shape_tri_count = len(tris_np)
        else:
            self.shape_tri_count = 0
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    # --------------------
    # Render helpers
    # --------------------
    def draw_points(self, color):
        if self.point_count == 0:
            return
        glColor3f(color[0], color[1], color[2])
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerf = glVertexPointer  # alias
        # tell GL where vertex data is
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        # draw all points
        glDrawArrays(GL_POINTS, 0, self.point_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def draw_lines(self, color):
        if not self.line_segments:
            return
        glColor3f(color[0], color[1], color[2])
        glLineWidth(cfg("LINE_THICKNESS"))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        # draw each segment using the offset parameter of glDrawArrays
        for (start, count) in self.line_segments:
            glDrawArrays(GL_LINE_STRIP, start, count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def draw_shapes(self):
        if self.shape_tri_count == 0:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_shapes)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        # shapes are stored as packed triangles; color is set per-shape by immediate draw above.
        glDrawArrays(GL_TRIANGLES, 0, self.shape_tri_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    # --------------------
    # Frame conversion: convert interpreter frame into contiguous numpy arrays and upload
    # --------------------
    def prepare_frame_gpu(self, frame):
        # Points
        points = frame.get("points", [])
        if points:
            pts_np = np.array([p["pos"] for p in points], dtype=np.float32)
        else:
            pts_np = np.zeros((0, 3), dtype=np.float32)
        self.upload_points(pts_np)
        
        # Lines: we flatten samples for all lines into single array and store segments
        lines = frame.get("lines", [])
        segments = []
        all_samples = []
        vertex_cursor = 0
        for ln in lines:
            samples = ln.get("samples", [])
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float32)
            all_samples.append(arr)
            count = arr.shape[0]
            segments.append((vertex_cursor, count))
            vertex_cursor += count
        if all_samples:
            lines_np = np.vstack(all_samples).astype(np.float32)
        else:
            lines_np = np.zeros((0, 3), dtype=np.float32)
        self.upload_lines(lines_np, segments)
        
        # Shapes: pack triangles sequentially, but we must set color per-triangle.
        # We'll prepare a single packed vertex buffer and keep a parallel color list for shapes.
        shapes = frame.get("shapes", [])
        triangles_list = []
        shapes_colors = []  # (start_idx, count, color) to draw separately if needed
        tri_cursor = 0
        for sh in shapes:
            tris = sh.get("triangles", [])
            color = sh.get("color", (1.0, 1.0, 1.0))
            if not tris:
                continue
            arr = np.array(tris, dtype=np.float32).reshape(-1, 3)  # Nx3 where N = 3*tri_count
            triangles_list.append(arr)
            count = arr.shape[0]
            shapes_colors.append((tri_cursor, count, color))
            tri_cursor += count
        if triangles_list:
            tris_np = np.vstack(triangles_list).astype(np.float32)
        else:
            tris_np = np.zeros((0, 3), dtype=np.float32)
        self.upload_shapes(tris_np)
        # store shapes_colors for per-shape color draws
        self._shapes_colors = shapes_colors
    
    # --------------------
    # Main render per frame
    # --------------------
    def render_frame(self, frame):
        # Prepare camera (fixed position)
        cam_pos = self.camera_position()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Fixed camera looking at origin
        gluLookAt(cam_pos[0], cam_pos[1], cam_pos[2],
                  0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0)
        # Draw points
        self.draw_points(cfg("POINT_COLOR"))
        # Draw lines (batches)
        self.draw_lines(cfg("LINE_COLOR"))
        # Draw shapes: since shapes can have different colors, we iterate shapes_colors and draw ranges
        if hasattr(self, "_shapes_colors") and self.shape_tri_count > 0:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_shapes)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
            for (start, count, color) in self._shapes_colors:
                glColor3f(color[0], color[1], color[2])
                glDrawArrays(GL_TRIANGLES, start, count)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        pygame.display.flip()
    
    # --------------------
    # Main loop
    # --------------------
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            # timing
            current_time = pygame.time.get_ticks()
            dt = current_time - self.last_time
            self.last_time = current_time
            if not self.paused:
                self.current_tick += dt
                # Update audio for the current time
                self.audio_player.update(self.current_tick)
            
            # read frame from interpreter
            frame = self.interpreter.read_full(self.current_tick)
            # convert and upload to GPU
            self.prepare_frame_gpu(frame)
            # render
            self.render_frame(frame)
            # debug caption
            pygame.display.set_caption(
                f"ZeroEngine | Time: {self.current_tick}ms | Points: {len(frame.get('points',[]))} | "
                f"Lines: {len(frame.get('lines',[]))} | Shapes: {len(frame.get('shapes',[]))} | "
                f"{'PAUSED' if self.paused else 'PLAYING'} | Audio: {'ON' if not self.paused else 'OFF'}"
            )
            self.clock.tick(self.target_fps)
        
        # Cleanup audio resources
        self.audio_player.cleanup()
        pygame.quit()

if __name__ == "__main__":
    engine = ZeroEngine("graphics.db")
    engine.run()