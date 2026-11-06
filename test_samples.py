import os
import sys
import pickle
import argparse
import numpy as np
from PIL import Image

# Ensure project imports resolve
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import IMAGES_DIRECTORY, CAPTIONS_PATH, MODEL_SAVE_PATH, TOKENIZER_PATH
from data_preprocessing import load_captions
from inference import greedy_generator, beam_search_generator


def get_feature_extractor():
    """Initializes InceptionV3 feature extractor."""
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.models import Model
    try:
        inception = InceptionV3(weights="imagenet", input_shape=(299, 299, 3))
    except Exception:
        inception = InceptionV3(weights=None, input_shape=(299, 299, 3))
    return Model(inputs=inception.inputs, outputs=inception.layers[-2].output)


def extract_features(cnn_model, image_path):
    """Loads and preprocesses an image for InceptionV3."""
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    img = Image.open(image_path).convert("RGB").resize((299, 299))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feats = cnn_model.predict(arr, verbose=0)
    return feats.flatten()


def run_tests(image_paths=None):
    """Runs caption prediction on a suite of test images."""
    print("=" * 75)
    print("      AI IMAGE CAPTION GENERATOR - TEST SUITE & BENCHMARK SAMPLES")
    print("=" * 75)

    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(TOKENIZER_PATH):
        print(f"[ERROR] Trained model or tokenizer missing.")
        print(f"Model path:     {MODEL_SAVE_PATH} (exists: {os.path.exists(MODEL_SAVE_PATH)})")
        print(f"Tokenizer path: {TOKENIZER_PATH} (exists: {os.path.exists(TOKENIZER_PATH)})")
        print("Please run 'python main.py' to generate model artifacts first.")
        return

    # Load captions dictionary
    captions_dict = {}
    if os.path.exists(CAPTIONS_PATH):
        print(f"[*] Loading human ground-truth captions from: {CAPTIONS_PATH}")
        raw_lines = load_captions(CAPTIONS_PATH)
        for line in raw_lines:
            parts = line.split(',', 1) if ',' in line else line.split('\t', 1)
            if len(parts) == 2:
                img_name, cap = parts[0].strip(), parts[1].strip()
                captions_dict.setdefault(img_name, []).append(cap)

    # Resolve test images
    if not image_paths:
        candidate_names = [
            "1000268201_693b08cb0e.jpg",  # Child climbing on playground
            "1001773457_577c3a7d70.jpg",  # Dog jumping/splashing in water
            "1002674143_1b742ab4b8.jpg",  # Young girl wearing yellow shirt
            "101669240_b2d3e7f17b.jpg",   # Brown dog walking along a path
            "1022420763_66c14944e8.jpg",  # Person standing on rock overlooking water
        ]
        image_paths = []
        for name in candidate_names:
            p = os.path.join(IMAGES_DIRECTORY, name)
            if os.path.exists(p):
                image_paths.append(p)

        # Fallback to any images found in IMAGES_DIRECTORY if specific filenames missing
        if not image_paths and os.path.exists(IMAGES_DIRECTORY):
            files = [f for f in os.listdir(IMAGES_DIRECTORY) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_paths = [os.path.join(IMAGES_DIRECTORY, f) for f in files[:5]]

    print(f"[*] Found {len(image_paths)} test image(s) to evaluate.")
    print("[*] Loading InceptionV3 CNN...")
    cnn_extractor = get_feature_extractor()

    print(f"[*] Loading Captioning Model from {MODEL_SAVE_PATH}...")
    from tensorflow.keras.models import load_model
    caption_model = load_model(MODEL_SAVE_PATH)

    print(f"[*] Loading Tokenizer from {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print("\n" + "=" * 75)
    results = []

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        print(f"\n[{idx}] Testing Image: {filename}")
        print(f"    Location: {img_path}")

        # Human ground-truth reference captions
        refs = captions_dict.get(filename, [])
        if refs:
            print("    Human Reference Captions:")
            for ref in refs[:3]:
                print(f"      - \"{ref}\"")

        # Feature extraction
        feats = extract_features(cnn_extractor, img_path)

        # Predictions
        cap_greedy = greedy_generator(caption_model, feats, tokenizer)
        cap_beam3 = beam_search_generator(caption_model, feats, tokenizer, K_beams=3)
        cap_beam5 = beam_search_generator(caption_model, feats, tokenizer, K_beams=5)

        print(f"    Predictions:")
        print(f"      - Greedy Search:     \"{cap_greedy}\"")
        print(f"      - Beam Search (K=3): \"{cap_beam3}\"")
        print(f"      - Beam Search (K=5): \"{cap_beam5}\"")

        results.append({
            "filename": filename,
            "path": img_path,
            "references": refs,
            "greedy": cap_greedy,
            "beam3": cap_beam3,
            "beam5": cap_beam5,
        })

    print("\n" + "=" * 75)
    print("                     ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 75)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Image Caption Generator on sample images")
    parser.add_argument("--image", type=str, default=None, help="Optional path to a custom image to test")
    args = parser.parse_args()

    custom_images = [args.image] if args.image else None
    run_tests(custom_images)
