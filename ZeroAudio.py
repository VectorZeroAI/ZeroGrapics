# ZeroAudio.py - Audio Player for ZeroGraphics

import sqlite3
import numpy as np
import pygame
import time
import pyttsx3  # For speech synthesis
import json
from typing import List, Dict, Any, Tuple, Optional

class ZeroAudioPlayer:
    def __init__(self, db_path: str, sample_rate: int = 44100):
        self.conn = sqlite3.connect(db_path)
        self.sample_rate = sample_rate
        self.audio = pygame.mixer
        self.audio.init(self.sample_rate, -16, 1, 512)
        self.sound_channels = {}
        self.pending_music_notes = []
        self.pending_speech = []
        self.played_speech_ids = set()
        self.speech_engine = None
        self.current_time_ms = 0
        self.paused = False
        self.last_update_time = time.time()
        
        # Initialize speech engine
        try:
            self.speech_engine = pyttsx3.init()
        except:
            print("Warning: Could not initialize speech engine. Speech functionality will be limited.")
        
        # Instrument mapping
        self.instruments = {
            "piano": self._generate_piano_note,
            "guitar": self._generate_guitar_note,
            "synth": self._generate_synth_note,
            # Add more instruments as needed
        }
        
        # Default instrument if none specified
        if "default" not in self.instruments:
            self.instruments["default"] = self._generate_synth_note
    
    def _generate_sine_wave(self, frequency: float, duration_ms: int, 
                           amplitude: float = 0.5, sample_rate: int = None) -> np.ndarray:
        """Generate a sine wave with proper fade-out to prevent clicks
        
        Args:
            frequency: Frequency in Hz
            duration_ms: Duration in milliseconds
            amplitude: Amplitude (0.0 to 1.0)
            sample_rate: Sample rate (defaults to self.sample_rate)
            
        Returns:
            Numpy array containing the audio samples
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # Calculate number of samples
        num_samples = int((duration_ms / 1000.0) * sample_rate)
        if num_samples <= 0:
            return np.array([], dtype=np.float32)
            
        # Generate time points
        t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
        
        # Generate sine wave
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Apply fade-out (10ms or proportional to note duration)
        fade_out_ms = min(10, duration_ms * 0.1)  # 10% of note duration, max 10ms
        fade_out_samples = int((fade_out_ms / 1000.0) * sample_rate)
        
        if fade_out_samples > 0 and len(wave) > 0:
            fade_curve = np.linspace(1.0, 0.0, min(fade_out_samples, len(wave)))
            wave[-len(fade_curve):] *= fade_curve
            
        return wave
    
    def _generate_piano_note(self, note_freq: float, duration_ms: int) -> np.ndarray:
        """Generate a piano-like note with ADSR envelope"""
        # Generate base sine wave
        wave = self._generate_sine_wave(note_freq, duration_ms, 0.4)
        
        # Apply ADSR envelope
        num_samples = len(wave)
        attack_samples = int(num_samples * 0.05)  # 5% attack
        decay_samples = int(num_samples * 0.2)    # 20% decay
        sustain_level = 0.7
        release_samples = int(num_samples * 0.2)  # 20% release
        
        envelope = np.ones(num_samples)
        
        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        if decay_samples > 0:
            envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain_level, decay_samples)
        
        # Sustain
        sustain_end = num_samples - release_samples
        if sustain_end > attack_samples + decay_samples:
            envelope[attack_samples+decay_samples:sustain_end] = sustain_level
        
        # Release
        if release_samples > 0 and sustain_end < num_samples:
            envelope[sustain_end:] = np.linspace(sustain_level, 0, num_samples - sustain_end)
        
        return wave * envelope
    
    def _generate_guitar_note(self, note_freq: float, duration_ms: int) -> np.ndarray:
        """Generate a guitar-like note with pluck simulation"""
        # Generate base wave with harmonics
        wave = self._generate_sine_wave(note_freq, duration_ms, 0.3)
        wave += self._generate_sine_wave(note_freq * 2, duration_ms, 0.2)
        wave += self._generate_sine_wave(note_freq * 3, duration_ms, 0.1)
        
        # Apply pluck envelope (fast decay)
        num_samples = len(wave)
        decay_samples = min(int(num_samples * 0.8), num_samples)
        
        if decay_samples > 0:
            envelope = np.exp(-np.linspace(0, 5, decay_samples))
            wave[:decay_samples] *= envelope
            
            if num_samples > decay_samples:
                wave[decay_samples:] = 0
        
        return wave
    
    def _generate_synth_note(self, note_freq: float, duration_ms: int) -> np.ndarray:
        """Generate a basic synth note"""
        return self._generate_sine_wave(note_freq, duration_ms, 0.5)
    
    def _note_to_frequency(self, note: str) -> float:
        """Convert note name (like 'C4') to frequency in Hz"""
        notes = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 
                 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        # Extract note name and octave
        if len(note) == 3 and note[1] == '#':
            note_name = note[:2]
            octave = int(note[2])
        else:
            note_name = note[0]
            octave = int(note[1])
        
        # Calculate semitones from A4 (440Hz)
        semitones = (octave - 4) * 12 + notes[note_name] - 9
        frequency = 440 * (2 ** (semitones / 12.0))
        return frequency
    
    def _create_sound(self, samples: np.ndarray) -> pygame.mixer.Sound:
        """Create a pygame sound object from audio samples"""
        # Ensure samples are in range [-1.0, 1.0]
        samples = np.clip(samples, -1.0, 1.0)
        
        # Convert to 16-bit integers
        int16_max = 32767
        sound_int16 = (samples * int16_max).astype(np.int16)
        
        # Create stereo sound (duplicate channels)
        stereo = np.column_stack((sound_int16, sound_int16))
        
        return self.audio.Sound(buffer=stereo.tobytes())
    
    def play_music_at_time(self, time_ms: int):
        """Play all music events that are active at the given time"""
        # Get all music events that start before or at the current time
        # and are still playing (start + duration > current time)
        cur = self.conn.cursor()
        cur.execute("""
            SELECT timestamp_ms, notes, durations, instrument_id 
            FROM music 
            WHERE timestamp_ms <= ? AND (timestamp_ms + CAST(json_extract(durations, '$[0]') AS INTEGER)) > ?
        """, (time_ms, time_ms))
        
        # Process each music event
        for row in cur.fetchall():
            try:
                timestamp_ms = row["timestamp_ms"]
                notes = json.loads(row["notes"])
                durations = json.loads(row["durations"])
                instrument_id = row["instrument_id"] or "default"
                
                # Get instrument generator function
                instrument_func = self.instruments.get(instrument_id, self.instruments["default"])
                
                # Play each note in the chord
                for i, note in enumerate(notes):
                    if i < len(durations):
                        duration = durations[i]
                        # Only play notes that are active at current time
                        if timestamp_ms + duration > time_ms:
                            freq = self._note_to_frequency(note)
                            # Generate sound for the remaining portion of the note
                            remaining_duration = min(duration, time_ms - timestamp_ms)
                            if remaining_duration > 0:
                                samples = instrument_func(freq, remaining_duration)
                                sound = self._create_sound(samples)
                                
                                # Play the sound
                                channel = sound.play()
                                if channel:
                                    self.sound_channels[channel] = {
                                        "end_time": time_ms + remaining_duration,
                                        "sound": sound
                                    }
            except Exception as e:
                print(f"Error processing music event at {timestamp_ms}: {e}")
    
    def play_scheduled_speech(self, current_time_ms: int):
        """Play speech that should start at or before the current time"""
        # Check if we need to start new speech
        if not self.pending_speech:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT id, sentence, start_time_ms, voice_id 
                FROM speech 
                WHERE start_time_ms <= ? 
                ORDER BY start_time_ms
            """, (current_time_ms,))
            
            for row in cur.fetchall():
                if row["id"] not in self.played_speech_ids:
                    self.pending_speech.append({
                        "id": row["id"],
                        "sentence": row["sentence"],
                        "start_time_ms": row["start_time_ms"],
                        "voice_id": row["voice_id"]
                    })
                    self.played_speech_ids.add(row["id"])
        
        # Process pending speech
        while self.pending_speech and self.speech_engine:
            speech = self.pending_speech.pop(0)
            self.speech_engine.say(speech["sentence"])
            self.speech_engine.runAndWait()  # Run and wait for THIS speech item
    
    def update(self, current_time_ms: int):
        """Update audio state based on current time"""
        if self.paused:
            return
            
        # Update time tracking
        self.current_time_ms = current_time_ms
        
        # Play music events for this time
        self.play_music_at_time(current_time_ms)
        
        # Play scheduled speech
        self.play_scheduled_speech(current_time_ms)
        
        # Clean up finished sounds
        current_time = time.time()
        channels_to_remove = []
        for channel, data in self.sound_channels.items():
            if data["end_time"] <= current_time_ms:
                channels_to_remove.append(channel)
        
        for channel in channels_to_remove:
            del self.sound_channels[channel]
    
    def pause(self):
        """Pause audio playback"""
        self.paused = True
        self.audio.pause()
    
    def play(self):
        """Resume audio playback"""
        self.paused = False
        self.audio.unpause()
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.speech_engine:
            self.speech_engine.stop()
        self.audio.quit()