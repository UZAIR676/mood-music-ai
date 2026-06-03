

DATASET_PATH = "maestro-v3.0.0"   # MIDI files  folder

# Token settings
PITCH_BINS    = 128
VELOCITY_BINS = 32
TIME_BINS     = 100
DURATION_BINS = 100

# Vocab size
VOCAB_SIZE = PITCH_BINS + VELOCITY_BINS + TIME_BINS + DURATION_BINS + 4

# Model size
SEQ_LEN  = 256
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
D_FF     = 512
DROPOUT  = 0.1

# Training
BATCH_SIZE = 8
EPOCHS     = 10
LR         = 3e-4
GRAD_CLIP  = 1.0

# Generation
TEMPERATURE = 1.0
TOP_K       = 50
GEN_LENGTH  = 512

# Paths
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "output"