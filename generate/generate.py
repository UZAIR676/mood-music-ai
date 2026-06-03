# ============================================================
# generate.py — Generate your own music!
# ============================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pretty_midi

from model.transformer import MusicTransformer
from train.config import *

MOODS = {
    "happy":     {"scale": [0,2,4,5,7,9,11], "temperature": 0.8,  "time_speed": 1.5},
    "sad":       {"scale": [0,2,3,5,7,8,10], "temperature": 0.6,  "time_speed": 0.6},
    "dramatic":  {"scale": [0,1,3,5,6,8,10], "temperature": 1.2,  "time_speed": 1.0},
    "relaxing":  {"scale": [0,2,4,7,9],       "temperature": 0.5,  "time_speed": 0.4},
}

TEMPOS = {
    "slow":   0.3,
    "medium": 0.6,
    "fast":   1.2,
}

def generate_music():
    device = torch.device("cpu")

    print("=" * 40)
    print("   Music AI - Generate Your Music!")
    print("=" * 40)

    # Mood
    print("\nChoose mood:")
    print("  happy / sad / dramatic / relaxing")
    mood_input = input("Mood: ").strip().lower()
    if mood_input not in MOODS:
        mood_input = "happy"
    mood = MOODS[mood_input]
    print(f"Mood: {mood_input}")

    # Tempo
    print("\nChoose tempo:")
    print("  slow / medium / fast")
    tempo_input = input("Tempo: ").strip().lower()
    if tempo_input not in TEMPOS:
        tempo_input = "medium"
    tempo = TEMPOS[tempo_input]
    print(f"Tempo: {tempo_input}")

    # Length
    print("\nHow many tokens to generate? (more = longer music)")
    print("  Recommended: 500=short  1000=medium  2000=long")
    try:
        gen_tokens = int(input("Tokens: ").strip())
        gen_tokens = max(200, min(5000, gen_tokens))
    except:
        gen_tokens = 1000
    print(f"Tokens: {gen_tokens}")

    # File name
    print("\nFile name? (press Enter for default)")
    name = input("Name: ").strip()
    if not name:
        name = f"{mood_input}_{tempo_input}"
    print(f"File: {name}.mid")

    # Load model
    print("\nLoading model...")
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    if not checkpoints:
        print("Train first: python main.py train")
        return
    latest = sorted(checkpoints)[-1]
    model  = MusicTransformer().to(device)
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, latest), map_location=device))
    model.eval()
    print(f"Loaded: {latest}")

    # Generate
    print(f"\nGenerating {gen_tokens} tokens...")
    scale  = mood["scale"]
    temp   = mood["temperature"]
    tokens = [60, 128, 160, 260]
    inp    = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        for i in range(gen_tokens):
            x      = inp[:, -SEQ_LEN:]
            logits = model(x)[:, -1, :] / temp
            top_k  = torch.topk(logits, TOP_K)
            probs  = torch.softmax(top_k.values, dim=-1)
            idx    = torch.multinomial(probs, 1)
            next_t = top_k.indices[0, idx[0, 0]]
            inp    = torch.cat([inp, next_t.view(1,1)], dim=1)
            tokens.append(next_t.item())

            if (i+1) % (gen_tokens // 4) == 0:
                print(f"  {((i+1)/gen_tokens*100):.0f}% done...")

    # Tokens to notes
    notes        = []
    current_time = 0.0
    i = 0
    while i + 3 < len(tokens):
        p = tokens[i]
        v = tokens[i+1] - PITCH_BINS
        t = tokens[i+2] - PITCH_BINS - VELOCITY_BINS
        d = tokens[i+3] - PITCH_BINS - VELOCITY_BINS - TIME_BINS

        if (0 <= p < PITCH_BINS and
            0 <= v < VELOCITY_BINS and
            0 <= t < TIME_BINS and
            0 <= d < DURATION_BINS):
            if p % 12 in scale:
                time_gap      = (t / 50.0) / max(0.1, tempo)
                current_time += time_gap
                notes.append({
                    "pitch":    p,
                    "velocity": max(1, v * 4),
                    "start":    current_time,
                    "end":      current_time + max(0.05, d / 50.0),
                })
        i += 4

    print(f"Total notes: {len(notes)}")
    print(f"Total duration: {current_time:.1f} seconds")

    if not notes:
        print("No notes generated, try again!")
        return

    # Save MIDI
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pm   = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for note in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=note["velocity"],
            pitch=note["pitch"],
            start=note["start"],
            end=note["end"],
        ))
    pm.instruments.append(inst)
    out_path = os.path.join(OUTPUT_DIR, f"{name}.mid")
    pm.write(out_path)

    print(f"\nMusic saved: {out_path}")
    print(f"Play it at: www.midiano.com")