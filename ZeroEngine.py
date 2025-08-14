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
    "SHAPE_FILL_COLOR": (0.5, 0.5, 1.0),
    "BACKGROUND_COLOR": (0.02, 0.02, 0.03),
    "WINDOW_SIZE": (1280, 720)
}

def cfg(name):
    return getattr(Config, name) if hasattr(Config, name) else DEFAULT_CONFIG[name]

class ZeroEngine:
    def __init__(self, db_path, width=None, height=None, sample_rate=60):
        pygame.init()
        self.width, self.height = width or cfg("WINDOW_SIZE")[0], height or cfg("WINDOW_SIZE")[1]
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("ZeroEngine")
        self.clock = pygame.time.Clock()
        self.target_fps = sample_rate
        
        # interpreter
        self.interpreter = ZeroInterpreter(db_path)
        self.audio_player = ZeroAudioPlayer(db_path)  # Always initialized
        
        # timing
        self.last_time = pygame.time.get_ticks()
        self.current_tick = 0
        self.paused = False
        
        # OpenGL setup
        self.setup_opengl()
        
        # GPU buffers
        self.vbo_points = None
        self.vbo_lines = None
        self.vbo_triangles = None
        self.point_count = 0
        self.line_count = 0
        self.tri_count = 0
        
        # Prepare initial frame
        self.prepare_frame_gpu(self.interpreter.read_full(0))
    
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
    
    # -# Camera utilities (simplified)# -
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
        return True
    
    def sample_line(self, p0, p1, num_samples):
        """Sample points along a line for rendering"""
        samples = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            z = p0[2] + t * (p1[2] - p0[2])
            samples.append([x, y, z])
        return samples
    
    def prepare_frame_gpu(self, frame):
        """Prepare frame data for GPU rendering"""
        # Handle points
        points = list(frame.get("points", {}).values())
        if points:
            self.point_array = np.array(points, dtype=np.float32)
            self.point_count = len(points)
        else:
            self.point_count = 0
        
        # Handle lines
        self.line_segments = []
        for line in frame.get("lines", []):
            p0 = line["points"][0]
            p1 = line["points"][1]
            self.line_segments.append((p0, p1))
        
        # Only process if we have line segments
        if self.line_segments:
            all_samples = []
            for p0, p1 in self.line_segments:
                samples = self.sample_line(p0, p1, 10)  # 10 samples per line
                if samples:
                    all_samples.append(samples)
            
            # Only vstack if we have samples
            if all_samples:
                self.line_vertices = np.vstack(all_samples)
                self.line_count = len(self.line_vertices)
            else:
                self.line_count = 0
        else:
            self.line_count = 0
        
        # Handle shapes (triangles)
        self.tri_vertices = []
        for shape in frame.get("shapes", []):
            # For each shape, triangulate the points
            coords = []
            for uuid in shape["points"]:
                if uuid in frame["points"]:
                    coords.append(frame["points"][uuid])
            
            # Triangulate the polygon (simple fan triangulation for convex polygons)
            if len(coords) >= 3:
                for i in range(1, len(coords) - 1):
                    # Create a triangle from the first point and two consecutive points
                    self.tri_vertices.append(coords[0])
                    self.tri_vertices.append(coords[i])
                    self.tri_vertices.append(coords[i + 1])
        
        # Convert to numpy array and set up for rendering
        if self.tri_vertices:
            self.tri_array = np.array(self.tri_vertices, dtype=np.float32)
            self.tri_count = len(self.tri_vertices)
        else:
            self.tri_count = 0
    
    def render_points(self):
        """Render points using VBOs"""
        if self.point_count == 0:
            return
            
        glColor3f(*cfg("POINT_COLOR"))
        
        # Create VBO if needed
        if self.vbo_points is None:
            self.vbo_points = glGenBuffers(1)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
        glBufferData(GL_ARRAY_BUFFER, self.point_array.nbytes, self.point_array, GL_STATIC_DRAW)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_POINTS, 0, self.point_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def render_lines(self):
        """Render lines using VBOs"""
        if self.line_count == 0:
            return
            
        glColor3f(*cfg("LINE_COLOR"))
        glLineWidth(cfg("LINE_THICKNESS"))
        
        # Create VBO if needed
        if self.vbo_lines is None:
            self.vbo_lines = glGenBuffers(1)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
        glBufferData(GL_ARRAY_BUFFER, self.line_vertices.nbytes, self.line_vertices, GL_STATIC_DRAW)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_LINE_STRIP, 0, self.line_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def render_shapes(self):
        """Render shapes using VBOs"""
        if self.tri_count == 0:
            return
            
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create VBO if needed
        if self.vbo_triangles is None:
            self.vbo_triangles = glGenBuffers(1)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_triangles)
        glBufferData(GL_ARRAY_BUFFER, self.tri_array.nbytes, self.tri_array, GL_STATIC_DRAW)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        # Draw all triangles
        glColor4f(0.5, 0.5, 1.0, 0.7)  # Blue with transparency
        glDrawArrays(GL_TRIANGLES, 0, self.tri_count)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisable(GL_BLEND)
    
    def render_frame(self, frame):
        """Render a complete frame"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set up camera
        camera_pos = self.camera_position()
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2],
                  0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0)
        
        # Render elements
        self.render_shapes()
        self.render_lines()
        self.render_points()
        
        pygame.display.flip()
    
    def run(self):
        """Main engine loop"""
        running = True
        while running:
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
            pygame.display.set_caption(f"ZeroEngine| Time: {self.current_tick}ms| Points: {len(frame.get('points',[]))}| Lines: {len(frame.get('lines',[]))}| Shapes: {len(frame.get('shapes',[]))}")
            
            # maintain frame rate
            self.clock.tick(self.target_fps)
            
            # handle events
            running = self.handle_events()
        
        # Cleanup
        self.audio_player.cleanup()
        pygame.quit()
        sys.exit()

def main():
    import argparse
    ap = argparse.ArgumentParser(description="ZeroEngine - Graphics and Audio Renderer")
    ap.add_argument("--db", default="graphics.db", help="Path to SQLite DB (default graphics.db)")
    ap.add_argument("--width", type=int, help="Window width")
    ap.add_argument("--height", type=int, help="Window height")
    args = ap.parse_args()
    
    if not os.path.exists(args.db):
        print("DB not found:", args.db)
        print("Run ZeroInit.py to create the DB first.")
        sys.exit(1)
    
    engine = ZeroEngine(args.db, args.width, args.height)
    engine.run()

if __name__ == "__main__":
    import os
    main()