"""
ZeroEngine.py - improved rendering for ZeroGraphics
Features:
 - Uses VBOs (glGenBuffers / glBufferData) for lines and triangles (shapes)
 - Renders points efficiently with GL_POINTS (fast) using a single vertex array
 - Reuses buffers between frames to avoid allocations
 - Camera: orbit (mouse drag), forward/back (scroll or W/S), pan (A/D, arrow keys)
 - Minimal immediate-mode usage (only for debug / fallback)
"""
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
        # camera params (spherical orbit)
        cam = np.array(cfg("CAMERA_POSITION"), dtype=np.float32)
        self.cam_distance = np.linalg.norm(cam)
        # angles
        self.cam_theta = 0.0  # horizontal angle
        self.cam_phi = math.radians(20.0)  # elevation angle
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # Input state
        self.mouse_orbiting = False
        self.last_mouse = (0, 0)
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
    # Camera utilities
    # --------------------
    def camera_position(self):
        # convert spherical to cartesian relative to camera_target
        r = self.cam_distance
        x = r * math.cos(self.cam_phi) * math.sin(self.cam_theta)
        y = r * math.sin(self.cam_phi)
        z = r * math.cos(self.cam_phi) * math.cos(self.cam_theta)
        return np.array([x, y, z], dtype=np.float32) + self.camera_target
    
    def handle_events(self):
        moved = False
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
                elif event.key == K_ESCAPE:
                    return False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # left click -> start orbit
                    self.mouse_orbiting = True
                    self.last_mouse = pygame.mouse.get_pos()
                elif event.button == 3:  # right click -> pan
                    self.last_mouse = pygame.mouse.get_pos()
                elif event.button == 4:  # wheel up -> zoom in
                    self.cam_distance = max(0.01, self.cam_distance * 0.9)
                elif event.button == 5:  # wheel down -> zoom out
                    self.cam_distance *= 1.1
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_orbiting = False
        # keyboard camera controls (W/S forward/back, A/D left/right, arrows pan)
        keys = pygame.key.get_pressed()
        move_speed = 0.05 * (self.cam_distance + 1.0)
        if keys[K_w] or keys[K_UP]:
            # move target forward along view direction
            cam_pos = self.camera_position()
            forward = (self.camera_target - cam_pos)
            forward = forward / (np.linalg.norm(forward) + 1e-9)
            self.camera_target += forward * move_speed
            moved = True
        if keys[K_s] or keys[K_DOWN]:
            cam_pos = self.camera_position()
            forward = (self.camera_target - cam_pos)
            forward = forward / (np.linalg.norm(forward) + 1e-9)
            self.camera_target -= forward * move_speed
            moved = True
        if keys[K_a]:
            # strafe left
            cam_pos = self.camera_position()
            forward = (self.camera_target - cam_pos)
            left = np.cross(self.camera_up, forward)
            left = left / (np.linalg.norm(left) + 1e-9)
            self.camera_target += left * move_speed
            moved = True
        if keys[K_d]:
            cam_pos = self.camera_position()
            forward = (self.camera_target - cam_pos)
            left = np.cross(self.camera_up, forward)
            left = left / (np.linalg.norm(left) + 1e-9)
            self.camera_target -= left * move_speed
            moved = True
        # mouse orbiting
        if self.mouse_orbiting:
            mx, my = pygame.mouse.get_pos()
            lx, ly = self.last_mouse
            dx = mx - lx
            dy = my - ly
            # rotate angles by pixels (tweak sensitivity)
            self.cam_theta += dx * 0.01
            self.cam_phi += -dy * 0.01
            # clamp phi avoid flipping
            self.cam_phi = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.cam_phi))
            self.last_mouse = (mx, my)
            moved = True
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
        else:
            # Ensure at least empty buffer of 12 bytes (3 floats for one vertex)
            glBufferData(GL_ARRAY_BUFFER, 12, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.line_segments = segments
    
    def upload_shapes(self, tris_np: np.ndarray):
        # tris_np: (T,3) float32 flattened triangles (T vertices)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_shapes)
        if len(tris_np) > 0:
            glBufferData(GL_ARRAY_BUFFER, tris_np.nbytes, tris_np, GL_DYNAMIC_DRAW)
            self.shape_tri_count = len(tris_np)
        else:
            glBufferData(GL_ARRAY_BUFFER, 12, None, GL_DYNAMIC_DRAW)
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
        glColor3f(color[0], color[1], color[2])
        glLineWidth(cfg("LINE_THICKNESS"))
        if not self.line_segments:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        # draw each segment individually using offsets
        for (start, count) in self.line_segments:
            # start is in vertex indices; pointer offset = start * 3 * 4 bytes
            ptr = ctypes.c_void_p(start * 3 * 4)
            glVertexPointer(3, GL_FLOAT, 0, ptr)
            glDrawArrays(GL_LINE_STRIP, 0, count)
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
        # Prepare camera
        cam_pos = self.camera_position()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(cam_pos[0], cam_pos[1], cam_pos[2],
                  self.camera_target[0], self.camera_target[1], self.camera_target[2],
                  self.camera_up[0], self.camera_up[1], self.camera_up[2])
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
                f"{'PAUSED' if self.paused else 'PLAYING'}"
            )
            self.clock.tick(self.target_fps)
        pygame.quit()

if __name__ == "__main__":
    engine = ZeroEngine("graphics.db")
    engine.run()