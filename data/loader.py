
import os
import pretty_midi

def get_all_midi_files(dataset_path):
    """Saari MIDI files ka list banao"""
    files = []
    for root, _, filenames in os.walk(dataset_path):
        for f in filenames:
            if f.endswith(".midi") or f.endswith(".mid"):
                files.append(os.path.join(root, f))
    return sorted(files)


def load_midi(filepath):
    """Ek MIDI file load karo aur notes return karo"""
    try:
        pm = pretty_midi.PrettyMIDI(filepath)
        if len(pm.instruments) == 0:
            return []
        instrument = pm.instruments[0]
        notes = [
            {
                "pitch":    note.pitch,
                "velocity": note.velocity,
                "start":    note.start,
                "end":      note.end,
            }
            for note in sorted(instrument.notes, key=lambda n: n.start)
        ]
        return notes
    except:
        return []


def explore_midi(filepath):
    """Ek file ka summary print karo"""
    pm    = pretty_midi.PrettyMIDI(filepath)
    inst  = pm.instruments[0]
    notes = inst.notes
    print(f"File     : {os.path.basename(filepath)}")
    print(f"Duration : {pm.get_end_time():.1f}s")
    print(f"Notes    : {len(notes)}")
    print(f"Pitch    : {min(n.pitch for n in notes)} - {max(n.pitch for n in notes)}")
    print(f"Velocity : {min(n.velocity for n in notes)} - {max(n.velocity for n in notes)}")
    print()