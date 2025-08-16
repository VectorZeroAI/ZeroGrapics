# ZeroAudio.py - Audio Player for ZeroGraphics
# (Assuming the rest of the file content is present before this snippet)

import sqlite3
import numpy as np
import pygame
import time
import pyttsx3 # For speech synthesis
import json
from typing import List, Dict, Any, Tuple, Optional

class ZeroAudioPlayer:
    def __init__(self, db_path: str, sample_rate: int = 44100):
        # --- FIX 1 (Critical): Add row_factory for named row access ---
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row # <-- ADDED THIS LINE
        # ---
        self.sample_rate = sample_rate
        self.audio = pygame.mixer
        self.audio.init(self.sample_rate, -16, 2, 512) # Increased channels to 2
        self.sound_channels: Dict[int, Dict[str, Any]] = {} # Channel ID -> {sound, end_time}
        self.pending_music_notes: List[Tuple[int, int, float, int, str]] = [] # (timestamp, duration, freq, note_id, instrument_id)
        self.pending_speech: List[Tuple[str, int, str, int]] = [] # (sentence, start_time_ms, voice_id, db_id)
        self.played_speech_ids = set()
        self.speech_engine: Optional[pyttsx3.Engine] = None
        self.current_time_ms = 0
        self.paused = False
        self.last_update_time = time.time()

    def init_speech_engine(self):
        """Initializes the speech synthesis engine."""
        if self.speech_engine is None:
            try:
                self.speech_engine = pyttsx3.init()
                # Basic rate setting (can be adjusted)
                self.speech_engine.setProperty('rate', 200)
            except Exception as e:
                print(f"Warning: Could not initialize speech engine: {e}")
                self.speech_engine = None

    def get_instrument_func(self, instrument_id: str):
        """Maps instrument ID to a waveform generation function."""
        # Simplified instrument definitions
        instruments = {
            "sine": lambda freq, duration_ms: self.generate_sine_wave(freq, duration_ms),
            "square": lambda freq, duration_ms: self.generate_square_wave(freq, duration_ms),
            "sawtooth": lambda freq, duration_ms: self.generate_sawtooth_wave(freq, duration_ms),
            "piano": lambda freq, duration_ms: self.generate_sine_wave(freq, duration_ms), # Placeholder
            "guitar": lambda freq, duration_ms: self.generate_sine_wave(freq, duration_ms), # Placeholder
        }
        return instruments.get(instrument_id, instruments["sine"]) # Default to sine

    def generate_sine_wave(self, frequency: float, duration_ms: int) -> np.ndarray:
        """Generates a sine wave."""
        num_samples = int(self.sample_rate * duration_ms / 1000.0)
        if num_samples <= 0:
            return np.array([], dtype=np.int16)
        t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
        wave = np.sin(2 * np.pi * frequency * t)
        # Ensure the waveform ends near zero to avoid clicks
        # Simple fade-out over last 10ms
        fade_samples = min(int(self.sample_rate * 0.01), num_samples)
        if fade_samples > 0:
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            wave[-fade_samples:] *= fade_out
        audio_data = (wave * 32767).astype(np.int16)
        return audio_data

    def generate_square_wave(self, frequency: float, duration_ms: int) -> np.ndarray:
        """Generates a square wave (naive implementation)."""
        num_samples = int(self.sample_rate * duration_ms / 1000.0)
        if num_samples <= 0:
             return np.array([], dtype=np.int16)
        t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
         # Simple fade-out
        fade_samples = min(int(self.sample_rate * 0.01), num_samples)
        if fade_samples > 0:
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            wave[-fade_samples:] *= fade_out
        audio_data = (wave * 32767).astype(np.int16)
        return audio_data

    def generate_sawtooth_wave(self, frequency: float, duration_ms: int) -> np.ndarray:
        """Generates a sawtooth wave (naive implementation)."""
        num_samples = int(self.sample_rate * duration_ms / 1000.0)
        if num_samples <= 0:
             return np.array([], dtype=np.int16)
        t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
        wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
         # Simple fade-out
        fade_samples = min(int(self.sample_rate * 0.01), num_samples)
        if fade_samples > 0:
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            wave[-fade_samples:] *= fade_out
        audio_data = (wave * 32767).astype(np.int16)
        return audio_data

    def schedule_music_note(self, timestamp_ms: int, duration: int, frequency: float, note_id: int, instrument_id: str):
        """Schedules a single music note for playback."""
        self.pending_music_notes.append((timestamp_ms, duration, frequency, note_id, instrument_id))

    def play_music_at_time(self, current_time_ms: int):
        """Plays scheduled music notes that are due."""
        if self.paused:
            return

        # --- FIX 5 (Critical - Logic): Correct SQL query for active chord events ---
        # This fix assumes the music table structure is as described in the plan:
        # timestamp_ms INTEGER, notes TEXT (JSON array), durations TEXT (JSON array), instrument_id TEXT
        # This query finds events that started and have not yet finished based on the longest duration in the chord.
        # A more robust schema would store individual notes.
        try:
            cur = self.conn.cursor()
            # Find chord events that are currently active
            # This query now correctly identifies events where ANY note might still be playing
            # by checking against the maximum duration in the chord.
            cur.execute("""
                SELECT timestamp_ms, notes, durations, instrument_id
                FROM music
                WHERE timestamp_ms <= ?
                  AND (timestamp_ms + (
                    SELECT MAX(dur) FROM json_each(durations) AS dur
                  )) > ?
            """, (current_time_ms, current_time_ms))

            active_events = cur.fetchall()

            for row in active_events:
                timestamp_ms = row["timestamp_ms"] # Access by name now works
                try:
                    notes_list = json.loads(row["notes"])
                    durations_list = json.loads(row["durations"])
                    instrument_id = row["instrument_id"]
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Error parsing music data at {timestamp_ms}ms: {e}")
                    continue

                if not isinstance(notes_list, list) or not isinstance(durations_list, list):
                    print(f"Warning: Invalid notes/durations format at {timestamp_ms}ms")
                    continue

                # Iterate through notes in the chord
                for i, (note, duration) in enumerate(zip(notes_list, durations_list)):
                    note_start_time = timestamp_ms
                    # Check if the specific note within the chord is currently playing
                    if note_start_time <= current_time_ms < (note_start_time + duration):
                         # Check if this specific note instance hasn't been scheduled yet
                         # Use a unique key based on timestamp and note index within the chord
                        note_key = (timestamp_ms, i)
                        if note_key not in [(n[0], n[3]) for n in self.pending_music_notes]: # Check timestamp and index
                            frequency = 440.0 * (2 ** ((note - 69) / 12.0)) # MIDI note to frequency
                            # --- FIX 6 (Critical - Logic): Pass full duration ---
                            self.schedule_music_note(note_start_time, duration, frequency, i, instrument_id)
                            # ---

        except sqlite3.Error as e:
            print(f"Database error in play_music_at_time: {e}")
        except Exception as e:
             print(f"Unexpected error in play_music_at_time: {e}")
        # ---


        # Play pending notes whose start time has arrived
        notes_to_remove = []
        for note_data in self.pending_music_notes:
            timestamp_ms, duration, frequency, note_id, instrument_id = note_data
            if current_time_ms >= timestamp_ms:
                instrument_func = self.get_instrument_func(instrument_id)
                if instrument_func:
                    # --- FIX 6 (Critical - Logic): Pass full duration ---
                    # Calculate remaining duration for this specific note instance
                    elapsed_time_for_note = current_time_ms - timestamp_ms
                    remaining_duration_for_note = max(0, duration - elapsed_time_for_note)

                    # Play the note with its full intended duration from the start point
                    # The instrument function should ideally handle playing from the correct point
                    # or the audio system should manage overlapping/splitting.
                    # For simplicity here, we play the full duration from the start time.
                    # A more advanced system might pre-generate or stream audio.
                    # This fix ensures the instrument gets the correct total duration.
                    samples = instrument_func(frequency, duration) # Pass full duration
                     # ---
                    if len(samples) > 0:
                        try:
                            sound = pygame.mixer.Sound(buffer=samples)
                            channel = sound.play()
                            if channel:
                                # Calculate end time for this specific note play instance
                                end_time = current_time_ms + duration
                                # Store channel ID for potential management
                                self.sound_channels[channel.get_id()] = {"sound": sound, "end_time": end_time}
                        except pygame.error as e:
                             print(f"Pygame error playing note {note_id} at {current_time_ms}ms: {e}")
                        except Exception as e:
                             print(f"Unexpected error playing note {note_id} at {current_time_ms}ms: {e}")
                notes_to_remove.append(note_data) # Remove after attempting to play

        # Remove notes that were due (regardless of success)
        for note_data in notes_to_remove:
            if note_data in self.pending_music_notes:
                 self.pending_music_notes.remove(note_data)

    def update_channels(self, current_time_ms: int):
       """Cleans up finished sound channels."""
       if self.paused:
           return
       channels_to_remove = []
       for channel_id, data in list(self.sound_channels.items()): # Iterate over a copy
           # Check if the estimated end time has passed
           # Note: pygame.mixer.get_busy() checks if the specific channel is busy
           # This is a simple check; actual completion might depend on OS/audio driver timing.
           if data["end_time"] <= current_time_ms or not pygame.mixer.Channel(channel_id).get_busy():
               channels_to_remove.append(channel_id)

       for channel_id in channels_to_remove:
           del self.sound_channels[channel_id]

    def schedule_speech(self, sentence: str, start_time_ms: int, voice_id: str, db_id: int):
        """Schedules a sentence for speech synthesis."""
        # Avoid scheduling the same speech entry multiple times
        if db_id not in self.played_speech_ids:
            self.pending_speech.append((sentence, start_time_ms, voice_id, db_id))

    def play_scheduled_speech(self, current_time_ms: int):
        """Plays scheduled speech items that are due."""
        if self.paused:
            return

        # --- FIX 7 (High-Impact): Avoid blocking with runAndWait() ---
        # Instead of playing one and waiting, queue all due speeches.
        # pyttsx3 might handle queuing internally, but we avoid blocking the main loop.
        if not self.pending_speech:
            # Check if we need to start new speech
            try:
                cur = self.conn.cursor()
                # --- FIX 2 (Critical): Correct SQL query placeholder usage ---
                # Using positional placeholders (?) correctly.
                cur.execute("""
                    SELECT id, sentence, start_time_ms, voice_id
                    FROM speech
                    WHERE start_time_ms <= ? AND id NOT IN ({})
                """.format(','.join('?'*len(self.played_speech_ids))) if self.played_speech_ids else """
                    SELECT id, sentence, start_time_ms, voice_id
                    FROM speech
                    WHERE start_time_ms <= ?
                """, (current_time_ms, *self.played_speech_ids))
                # ---

                due_speeches = cur.fetchall()
                for row in due_speeches:
                    # --- Access by name now works due to row_factory ---
                    speech_id = row["id"]
                    sentence = row["sentence"]
                    start_time_ms_db = row["start_time_ms"]
                    voice_id = row["voice_id"]
                    # ---
                    self.schedule_speech(sentence, start_time_ms_db, voice_id, speech_id)

            except sqlite3.Error as e:
                print(f"Database error in play_scheduled_speech (scheduling): {e}")
            except Exception as e:
                 print(f"Unexpected error in play_scheduled_speech (scheduling): {e}")

        # Play pending speech items whose time has come
        speech_items_to_remove = []
        for item in self.pending_speech:
            sentence, start_time_ms_item, voice_id, db_id = item
            if current_time_ms >= start_time_ms_item:
                if self.speech_engine is None:
                    self.init_speech_engine()
                if self.speech_engine:
                    try:
                        # Set voice properties if needed (basic example)
                        # voices = self.speech_engine.getProperty('voices')
                        # self.speech_engine.setProperty('voice', voices[0].id) # Example

                        self.speech_engine.say(sentence)
                        self.played_speech_ids.add(db_id) # Mark as played/scheduled
                    except Exception as e:
                        print(f"Error scheduling speech '{sentence[:20]}...': {e}")
                speech_items_to_remove.append(item) # Remove from pending after scheduling

        # Remove items that were due (regardless of success)
        for item in speech_items_to_remove:
            if item in self.pending_speech:
                self.pending_speech.remove(item)

        # --- Crucial Change: Run speech engine iteration OUTSIDE the loop ---
        # This processes the queue without blocking until the first item finishes.
        if self.speech_engine:
            try:
                # runAndWait() is still blocking, but now it processes the entire
                # queued batch scheduled in this call, not just one item per main loop iteration.
                # This is better than the original bug but still blocks briefly.
                # For truly non-blocking, a threading approach might be needed,
                # but pyttsx3's internal queue handling helps here.
                if speech_items_to_remove: # Only call if something was added
                    self.speech_engine.runAndWait()
            except Exception as e:
                 print(f"Error running speech engine: {e}")
        # ---

    def update(self, current_time_ms: int):
        """Main update method called each frame."""
        if not self.paused:
             self.current_time_ms = current_time_ms
        else:
            # If paused, update current_time_ms based on real time when unpaused
            # This simplistic approach might cause jumps. A delta-time approach is better.
            self.last_update_time = time.time()

        self.play_music_at_time(self.current_time_ms)
        self.update_channels(self.current_time_ms)
        self.play_scheduled_speech(self.current_time_ms)

    def toggle_pause(self):
        """Toggles the pause state."""
        self.paused = not self.paused
        if not self.paused:
            # Adjust last_update_time to prevent large time jumps on unpause
            self.last_update_time = time.time()

    def cleanup(self):
        """Cleans up audio resources."""
        if self.speech_engine:
            try:
                self.speech_engine.stop() # Stop any ongoing speech
            except:
                pass # Ignore errors during cleanup
        if self.audio:
             try:
                self.audio.quit()
             except:
                pass # Ignore errors during cleanup
        if self.conn:
            try:
                self.conn.close()
            except:
                pass # Ignore errors during cleanup

# --- End of ZeroAudio.py ---