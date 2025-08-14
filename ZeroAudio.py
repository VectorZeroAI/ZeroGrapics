"""
ZeroAudio.py
- Handles audio playback in sync with the video
- Reads from the music and speech tables in the database
- Generates and plays audio using pygame.mixer and a TTS engine
"""

import pygame
import sqlite3
import json
import time
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import pyttsx3  # For text-to-speech

class ZeroAudioPlayer:
    def __init__(self, db_path: str, sample_rate: int = 44100):
        """
        Initialize the audio player
        
        Args:
            db_path: Path to the SQLite database
            sample_rate: Audio sample rate (default: 44100 Hz)
        """
        # Initialize pygame mixer
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
        
        # Connect to database
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.current_time = 0  # Current time in ms
        self.is_playing = False
        self.speech_engine = None
        self.pending_speech = []  # Queue of speech to be played
        self.last_music_time = -1  # Last time we processed music
        
        # Initialize TTS engine
        self._init_tts_engine()
        
        # Load instrument sounds
        self.instruments = {}
        self._load_instruments()
        
    def _init_tts_engine(self):
        """Initialize the text-to-speech engine"""
        try:
            self.speech_engine = pyttsx3.init()
            # Configure voice properties
            self.speech_engine.setProperty('rate', 150)  # words per minute
            self.speech_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Warning: Could not initialize TTS engine: {e}")
            self.speech_engine = None
            
    def _load_instruments(self):
        """
        Load instrument sounds.
        In a real implementation, this would load actual sound samples or
        synthesize sounds based on the instrument ID.
        """
        # For demonstration, we'll just define some basic wave generation functions
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
        """
        Generate a sine wave
        
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
        
        # Generate time values
        t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
        
        # Generate sine wave
        wave = amplitude * np.sin(frequency * t * 2 * np.pi)
        
        # Apply fade-out to avoid clicks
        fade_out = np.linspace(1.0, 0.0, int(sample_rate * 0.01))  # 10ms fade out
        if len(wave) > len(fade_out):
            wave[-len(fade_out):] *= fade_out
        
        return wave
    
    def _generate_piano_note(self, note: int, duration_ms: int) -> np.ndarray:
        """
        Generate a piano-like note
        
        Args:
            note: MIDI note number (0-127)
            duration_ms: Duration in milliseconds
            
        Returns:
            Numpy array containing the audio samples
        """
        # Convert MIDI note to frequency
        frequency = 440 * (2 ** ((note - 69) / 12.0))
        
        # Generate the main sine wave
        wave = self._generate_sine_wave(frequency, duration_ms, 0.3)
        
        # Add some harmonics to make it sound more piano-like
        harmonic2 = self._generate_sine_wave(frequency * 2, duration_ms, 0.2)
        harmonic3 = self._generate_sine_wave(frequency * 3, duration_ms, 0.1)
        
        # Combine the waves
        if len(harmonic2) > len(wave):
            harmonic2 = harmonic2[:len(wave)]
        if len(harmonic3) > len(wave):
            harmonic3 = harmonic3[:len(wave)]
            
        wave[:len(harmonic2)] += harmonic2
        wave[:len(harmonic3)] += harmonic3
        
        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave /= max_val
            
        return wave
    
    def _generate_guitar_note(self, note: int, duration_ms: int) -> np.ndarray:
        """
        Generate a guitar-like note
        
        Args:
            note: MIDI note number (0-127)
            duration_ms: Duration in milliseconds
            
        Returns:
            Numpy array containing the audio samples
        """
        frequency = 440 * (2 ** ((note - 69) / 12.0))
        
        # Generate a slightly detuned sine wave for a richer sound
        wave1 = self._generate_sine_wave(frequency, duration_ms, 0.25)
        wave2 = self._generate_sine_wave(frequency * 1.005, duration_ms, 0.25)
        
        # Combine
        wave = wave1 + wave2
        
        # Apply an envelope to simulate pluck decay
        num_samples = len(wave)
        envelope = np.exp(-np.linspace(0, 5, num_samples))  # Exponential decay
        wave *= envelope
        
        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave /= max_val
            
        return wave
    
    def _generate_synth_note(self, note: int, duration_ms: int) -> np.ndarray:
        """
        Generate a synth-like note
        
        Args:
            note: MIDI note number (0-127)
            duration_ms: Duration in milliseconds
            
        Returns:
            Numpy array containing the audio samples
        """
        frequency = 440 * (2 ** ((note - 69) / 12.0))
        
        # Generate a square wave
        num_samples = int((duration_ms / 1000.0) * self.sample_rate)
        t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
        wave = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
        
        # Add some noise for texture
        noise = 0.05 * np.random.normal(0, 1, num_samples)
        wave += noise
        
        # Apply ADSR envelope
        attack = int(0.01 * num_samples)  # 10ms attack
        decay = int(0.1 * num_samples)    # 100ms decay
        sustain = 0.7
        release = int(0.05 * num_samples) # 50ms release
        
        envelope = np.ones(num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, sustain, decay)
        envelope[attack+decay:num_samples-release] = sustain
        envelope[num_samples-release:] = np.linspace(sustain, 0, release)
        
        wave *= envelope
        
        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave /= max_val
            
        return wave
    
    def _play_sound(self, sound_array: np.ndarray):
        """
        Play a sound from a numpy array
        
        Args:
            sound_array: Numpy array containing audio samples
        """
        # Convert to 16-bit signed integers
        sound_int16 = (sound_array * 32767).astype(np.int16)
        
        # Create a stereo sound (duplicate mono to both channels)
        stereo_sound = np.column_stack((sound_int16, sound_int16))
        
        # Create a Sound object and play it
        sound = pygame.sndarray.make_sound(stereo_sound)
        sound.play()
    
    def play_music_at_time(self, time_ms: int):
        """
        Play music that should start at the given time
        
        Args:
            time_ms: Current time in milliseconds
        """
        # Only process each ms once
        if time_ms <= self.last_music_time:
            return
            
        self.last_music_time = time_ms
        
        # Get music data for this specific time
        cur = self.conn.cursor()
        cur.execute("SELECT notes, durations, instrument_id FROM music WHERE timestamp_ms = ?", (time_ms,))
        row = cur.fetchone()
        
        if not row:
            return
            
        notes_json, durations_json, instrument_id = row
        
        try:
            notes = json.loads(notes_json)
            durations = json.loads(durations_json)
        except json.JSONDecodeError:
            return
            
        # Get the instrument function
        instrument_func = self.instruments.get(instrument_id, self.instruments["default"])
        
        # Play each note
        for i in range(min(len(notes), len(durations))):
            note = notes[i]
            duration = durations[i]
            
            # Generate the sound
            sound_array = instrument_func(note, duration)
            
            # Play the sound
            self._play_sound(sound_array)
    
    def schedule_speech_at_time(self, time_ms: int):
        """
        Schedule speech that should start at or before the given time
        
        Args:
            time_ms: Current time in milliseconds
        """
        # Check for speech that should start now or earlier but hasn't been scheduled yet
        cur = self.conn.cursor()
        cur.execute("SELECT sentence, voice_id FROM speech WHERE start_time_ms <= ? AND start_time_ms > ?", 
                   (time_ms, self.current_time))
        rows = cur.fetchall()
        
        for row in rows:
            sentence, voice_id = row
            self.pending_speech.append((sentence, voice_id))
    
    def play_scheduled_speech(self):
        """Play any scheduled speech"""
        while self.pending_speech and self.speech_engine:
            sentence, voice_id = self.pending_speech.pop(0)
            
            # Configure voice based on voice_id
            if "female" in voice_id.lower():
                # This is platform-dependent; might need to select a specific voice
                voices = self.speech_engine.getProperty('voices')
                for voice in voices:
                    if "female" in voice.name.lower():
                        self.speech_engine.setProperty('voice', voice.id)
                        break
            
            # Queue the sentence for speaking
            self.speech_engine.say(sentence)
        
        # Run the TTS engine to process the queue
        if self.pending_speech and self.speech_engine:
            self.speech_engine.runAndWait()
    
    def update(self, time_ms: int):
        """
        Update the audio player for the current time
        
        Args:
            time_ms: Current time in milliseconds
        """
        self.current_time = time_ms
        
        # Play music for this time
        self.play_music_at_time(time_ms)
        
        # Schedule speech for this time
        self.schedule_speech_at_time(time_ms)
        
        # Play any scheduled speech
        self.play_scheduled_speech()
    
    def play(self):
        """Start audio playback"""
        self.is_playing = True
    
    def pause(self):
        """Pause audio playback"""
        self.is_playing = False
        pygame.mixer.pause()
        if self.speech_engine:
            self.speech_engine.stop()
    
    def stop(self):
        """Stop audio playback"""
        self.is_playing = False
        pygame.mixer.stop()
        if self.speech_engine:
            self.speech_engine.stop()
            self.speech_engine.reset()
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        pygame.mixer.quit()
        if self.speech_engine:
            self.speech_engine.stop()
            self.speech_engine = None
        self.conn.close()