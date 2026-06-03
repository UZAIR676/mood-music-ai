
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.config import *

def notes_to_tokens(notes):
    """
    Har note ko 4 tokens mein badlo:
    1. Pitch   — kaun sa note (0-127)
    2. Velocity — kitna zor se (0-31)
    3. Time    — pichle note ke baad kitni der (0-99)
    4. Duration — note kitna lamba (0-99)
    """
    tokens = []
    prev_start = 0.0

    for note in notes:
        # Pitch token
        pitch = note["pitch"]  # 0-127

        # Velocity token 
        velocity = min(note["velocity"] // 4, VELOCITY_BINS - 1)

        # Time token 
        time_diff = note["start"] - prev_start
        time_bin = min(int(time_diff * 50), TIME_BINS - 1)

        # Duration token 
        duration = note["end"] - note["start"]
        dur_bin = min(int(duration * 50), DURATION_BINS - 1)

        tokens.append(pitch)                              # 0-127
        tokens.append(PITCH_BINS + velocity)              # 128-159
        tokens.append(PITCH_BINS + VELOCITY_BINS + time_bin)        # 160-259
        tokens.append(PITCH_BINS + VELOCITY_BINS + TIME_BINS + dur_bin)  # 260-359

        prev_start = note["start"]

    return tokens


def tokens_to_notes(tokens):
    """
    Numbers ko wapas music notes mein badlo
    """
    notes = []
    current_time = 0.0

    i = 0
    while i + 3 < len(tokens):
        pitch    = tokens[i]
        velocity = (tokens[i+1] - PITCH_BINS) * 4
        time_bin = tokens[i+2] - PITCH_BINS - VELOCITY_BINS
        dur_bin  = tokens[i+3] - PITCH_BINS - VELOCITY_BINS - TIME_BINS

        # Ranges check karo
        if not (0 <= pitch < PITCH_BINS): break
        if not (0 <= velocity < 128):     break

        time_diff = time_bin / 50.0
        duration  = dur_bin  / 50.0

        current_time += time_diff

        notes.append({
            "pitch":    pitch,
            "velocity": max(1, min(127, velocity)),
            "start":    current_time,
            "end":      current_time + max(0.05, duration),
        })

        i += 4

    return notes