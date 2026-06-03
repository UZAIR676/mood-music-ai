
import sys

def main():
    if len(sys.argv) < 2:
        print("=" * 40)
        print("  Music AI - Commands:")
        print("=" * 40)
        print("  python main.py train     — Model sikhao")
        print("  python main.py generate  — Music banao")
        print("  python main.py explore   — Data dekho")
        print("=" * 40)
        return

    mode = sys.argv[1]

    if mode == "train":
        from train.train import train
        train()

    elif mode == "generate":
        from generate.generate import generate_music
        generate_music()

    elif mode == "explore":
        from data.loader import get_all_midi_files, explore_midi
        from train.config import DATASET_PATH
        files = get_all_midi_files(DATASET_PATH)
        print(f"Total files: {len(files)}")
        print()
        for f in files[:3]:
            explore_midi(f)

    else:
        print(f"Unknown command: {mode}")
        print("Use: train, generate, ya explore")


if __name__ == "__main__":
    main()