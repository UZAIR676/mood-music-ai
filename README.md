# 🎹 maestro-ai

> A transformer-based AI that generates piano music from scratch — trained on the MAESTRO dataset with mood and tempo control.

---

## What is this?

maestro-ai is a music generation model that learns patterns from professional piano recordings and creates original MIDI music based on your chosen **mood** and **tempo**.

Think of it like GPT — but instead of predicting the next word, it predicts the next musical note.

---

## Features

- 🎵 **Generates original piano music** — no loops, no samples, pure AI composition
- 🎭 **4 mood modes** — Happy, Sad, Dramatic, Relaxing (each uses a different musical scale)
- ⚡ **3 tempo modes** — Slow, Medium, Fast
- 🎛️ **Adjustable length** — 200 to 2000 tokens
- 🌐 **Web interface** — clean browser UI to generate and download MIDI files
- 💾 **MIDI output** — play your generated music on any MIDI player or DAW

---

## How it works

Each note is converted into **4 tokens**:

```
[Pitch] [Velocity] [Time gap] [Duration]
  0-127   128-159    160-259    260-359
```

The transformer model learns from thousands of these token sequences and generates new ones. After generation, notes are filtered by the mood's musical scale:

| Mood | Scale | Temperature |
|------|-------|-------------|
| Happy | Major | 0.8 |
| Sad | Minor | 0.6 |
| Dramatic | Diminished | 1.2 |
| Relaxing | Pentatonic | 0.5 |

---

## Demo

Generate music → download `.mid` → play at [midiano.com](https://midiano.com)

```
Mood: Happy  |  Tempo: Medium  |  Tokens: 1000
→ ~250 notes  |  ~1.5 minutes of music
```

---

## Installation

```bash
git clone https://github.com/your-username/maestro-ai
cd maestro-ai
pip install -r requirements.txt
```

**Requirements:**
```
torch
pretty_midi
flask
```

---

## Usage

### Train the model

```bash
python main.py train
```

Checkpoints are saved automatically after each epoch in `/checkpoints`.

### Generate music (CLI)

```bash
python main.py generate
```

Follow the prompts to choose mood, tempo, and length.

### Web interface

```bash
python app2.py
```

Open `http://localhost:5000` in your browser.

---

## Project structure

```
maestro-ai/
├── app2.py                  # Flask web app
├── main.py                  # CLI entry point
├── model/
│   └── transformer.py       # MusicTransformer model
├── train/
│   └── config.py            # Hyperparameters & settings
├── data/
│   ├── loader.py            # MIDI file loader
│   └── tokenizer.py         # Note → token converter
├── generate/
│   └── generate.py          # Music generation logic
├── checkpoints/             # Saved model weights
├── maestro-v3.0.0/          # Training dataset (MIDI files)
└── templates/
    └── index.html           # Web UI
```

---

## Dataset

Trained on the [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) dataset by Google Magenta — over 200 hours of professional piano recordings.

---

## Token estimation

| Tokens | Notes | Slow | Medium | Fast |
|--------|-------|------|--------|------|
| 500 | ~100 | ~3 min | ~45 sec | ~20 sec |
| 1000 | ~250 | ~5 min | ~1.5 min | ~45 sec |
| 2000 | ~500 | ~10 min | ~3 min | ~1.5 min |

*(Approximate — varies by mood and scale filtering)*

---

## Built with

- [PyTorch](https://pytorch.org/) — transformer model
- [pretty_midi](https://craffel.github.io/pretty-midi/) — MIDI processing
- [Flask](https://flask.palletsprojects.com/) — web interface
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) — training data

---

## License

MIT