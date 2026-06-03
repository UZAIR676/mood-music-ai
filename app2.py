# ============================================================
# app2.py — Web interface for Music AI
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, send_file, jsonify
import torch
import pretty_midi

from model.transformer import MusicTransformer
from train.config import *

app = Flask(__name__)

MOODS = {
    "happy":    {"scale": [0,2,4,5,7,9,11], "temperature": 0.8},
    "sad":      {"scale": [0,2,3,5,7,8,10], "temperature": 0.6},
    "dramatic": {"scale": [0,1,3,5,6,8,10], "temperature": 1.2},
    "relaxing": {"scale": [0,2,4,7,9],      "temperature": 0.5},
}

TEMPOS = {
    "slow":   0.3,
    "medium": 0.6,
    "fast":   1.2,
}

def load_model():
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    if not checkpoints:
        return None
    latest = sorted(checkpoints)[-1]
    model  = MusicTransformer()
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, latest), map_location="cpu"))
    model.eval()
    return model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data       = request.json
    mood_input = data.get("mood", "happy")
    tempo_input= data.get("tempo", "medium")
    gen_tokens = int(data.get("tokens", 1000))
    name       = data.get("name", "my_music").strip() or "my_music"

    mood  = MOODS.get(mood_input, MOODS["happy"])
    tempo = TEMPOS.get(tempo_input, TEMPOS["medium"])

    model = load_model()
    if not model:
        return jsonify({"error": "No model found! Train first."}), 400

    # Generate tokens
    tokens = [60, 128, 160, 260]
    inp    = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        for _ in range(gen_tokens):
            x      = inp[:, -SEQ_LEN:]
            logits = model(x)[:, -1, :] / mood["temperature"]
            top_k  = torch.topk(logits, TOP_K)
            probs  = torch.softmax(top_k.values, dim=-1)
            idx    = torch.multinomial(probs, 1)
            next_t = top_k.indices[0, idx[0, 0]]
            inp    = torch.cat([inp, next_t.view(1,1)], dim=1)
            tokens.append(next_t.item())

    # Tokens to notes
    notes        = []
    current_time = 0.0
    i = 0
    while i + 3 < len(tokens):
        p = tokens[i]
        v = tokens[i+1] - PITCH_BINS
        t = tokens[i+2] - PITCH_BINS - VELOCITY_BINS
        d = tokens[i+3] - PITCH_BINS - VELOCITY_BINS - TIME_BINS

        if (0 <= p < PITCH_BINS and 0 <= v < VELOCITY_BINS and
            0 <= t < TIME_BINS  and 0 <= d < DURATION_BINS):
            if p % 12 in mood["scale"]:
                current_time += (t / 50.0) / max(0.1, tempo)
                notes.append({
                    "pitch":    p,
                    "velocity": max(1, v * 4),
                    "start":    current_time,
                    "end":      current_time + max(0.05, d / 50.0),
                })
        i += 4

    if not notes:
        return jsonify({"error": "No notes generated, try again!"}), 400

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

    return jsonify({
        "success": True,
        "notes":    len(notes),
        "duration": round(current_time, 1),
        "file":     f"{name}.mid"
    })

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    print("Opening Music AI at: http://localhost:5000")
    app.run(debug=True)