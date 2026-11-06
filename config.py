import os

# --- Path Configurations ---
# Automatically support Kaggle environment, Google Colab, or local directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KAGGLE_IMG_DIR = "/kaggle/input/flickr8k/Images/"
KAGGLE_CAP_PATH = "/kaggle/input/flickr8k/captions.txt"

# Default local data directories
LOCAL_IMG_DIR = os.path.join(BASE_DIR, "data", "Images")
LOCAL_CAP_PATH = os.path.join(BASE_DIR, "data", "captions.txt")

def resolve_dataset_paths():
    """Dynamically resolves images directory and captions file path, preferring full Flickr8k if available."""
    # 1. Native Kaggle environment
    if os.path.exists(KAGGLE_IMG_DIR) and os.path.exists(KAGGLE_CAP_PATH):
        return KAGGLE_IMG_DIR, KAGGLE_CAP_PATH

    # 2. Check for full dataset in kagglehub cache
    try:
        cache_base = os.path.expanduser("~/.cache/kagglehub/datasets/adityajn105/flickr8k/versions")
        if os.path.exists(cache_base):
            versions = sorted(os.listdir(cache_base))
            if versions:
                v_dir = os.path.join(cache_base, versions[-1])
                img_dir = os.path.join(v_dir, "Images")
                cap_file = os.path.join(v_dir, "captions.txt")
                if os.path.exists(img_dir) and os.path.exists(cap_file):
                    return img_dir, cap_file
    except Exception:
        pass

    # 3. Local data directory
    if os.path.exists(LOCAL_IMG_DIR) and os.path.exists(LOCAL_CAP_PATH):
        return LOCAL_IMG_DIR, LOCAL_CAP_PATH

    # Default to local data directory
    return LOCAL_IMG_DIR, LOCAL_CAP_PATH

IMAGES_DIRECTORY, CAPTIONS_PATH = resolve_dataset_paths()

# Built-in Default Images (Zero input required)
DEFAULT_SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")
DEFAULT_IMAGE_PATH = os.path.join(DEFAULT_SAMPLES_DIR, "1001773457_577c3a7d70.jpg")

# Output cache & model artifacts
FEATURES_CACHE_PATH = os.path.join(BASE_DIR, "image_features.pkl")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "caption_model.keras")

# --- Model & Hyperparameter Settings ---
IMAGE_SIZE = (299, 299)
CNN_OUTPUT_DIM = 2048 # InceptionV3 penultimate layer output dimension
EMBEDDING_DIM = 256
LSTM_UNITS = 256
DENSE_UNITS = 256

BATCH_SIZE_TRAIN = 270
BATCH_SIZE_VAL = 150
EPOCHS = 15
LEARNING_RATE = 0.01
CLIP_NORM = 1.0

RANDOM_SEED = 42
TEST_SIZE_1 = 0.15 # 15% validation + test
TEST_SIZE_2 = 0.10 # 10% test from val

if __name__ == "__main__":
    print("=== Configuration Status ===")
    print(f"Base Directory:     {BASE_DIR}")
    print(f"Images Directory:   {IMAGES_DIRECTORY} (Exists: {os.path.exists(IMAGES_DIRECTORY)})")
    print(f"Captions Path:      {CAPTIONS_PATH} (Exists: {os.path.exists(CAPTIONS_PATH)})")
    print(f"Model Save Path:    {MODEL_SAVE_PATH} (Exists: {os.path.exists(MODEL_SAVE_PATH)})")
    print(f"Tokenizer Path:     {TOKENIZER_PATH} (Exists: {os.path.exists(TOKENIZER_PATH)})")
    print(f"Features Cache:     {FEATURES_CACHE_PATH} (Exists: {os.path.exists(FEATURES_CACHE_PATH)})")

