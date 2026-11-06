import argparse
import os
import sys
import pickle

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tensorflow.keras.models import load_model

from config import (
    IMAGES_DIRECTORY,
    CAPTIONS_PATH,
    FEATURES_CACHE_PATH,
    TOKENIZER_PATH,
    MODEL_SAVE_PATH,
    CNN_OUTPUT_DIM
)
from data_preprocessing import prepare_dataset
from extract_features import build_inception_extractor, extract_and_split_features
from train import train
from inference import greedy_generator, beam_search_generator, caption_image_file
from evaluate import visualaization, captions_length, visualization, BLEU_score


def run_pipeline(beam_size=3, image_path=None):
    """
    Executes the complete end-to-end pipeline automatically with zero input required:
    1. Dataset loading & summary
    2. Exploratory visualizations (histogram & samples)
    3. Model verification & loading (trains if missing)
    4. Test evaluation with BLEU-1 and BLEU-2 scoring
    5. Inference with Greedy and Beam Search on sample images
    """
    print("\n" + "=" * 65)
    print("FLICKR8K IMAGE CAPTION GENERATOR - ZERO-INPUT PIPELINE")
    print("=" * 65)

    # 1. Dataset loading
    print("\n--- Step 1: Loading & Inspecting Dataset ---")
    dataset = prepare_dataset(CAPTIONS_PATH, IMAGES_DIRECTORY)
    print(f"* Images Directory:   {IMAGES_DIRECTORY}")
    print(f"* Captions Path:      {CAPTIONS_PATH}")
    print(f"* Total Images:       {len(dataset['all_image_ids'])}")
    print(f"* Total Captions:     {len(dataset['raw_captions'])}")
    print(f"* Vocabulary Size:    {dataset['vocab_size']}")
    print(f"* Max Caption Length: {dataset['max_caption_length']}")
    print(f"* Train / Val / Test: {len(dataset['train_image_ids'])} / {len(dataset['val_image_ids'])} / {len(dataset['test_image_ids'])} images")

    # 2. EDA Visualizations
    print("\n--- Step 2: Generating Exploratory Visualizations ---")
    captions_length(dataset["cleaned_captions"], save_path="captions_histogram.png")
    visualaization(dataset["captions_ids"], 4, IMAGES_DIRECTORY, save_path="eda_samples.png")

    # 3. Model Verification / Training
    print("\n--- Step 3: Verifying Trained Model & Tokenizer ---")
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(TOKENIZER_PATH):
        print(f"Trained model not found at {MODEL_SAVE_PATH}. Starting automated training...")
        train()

    print(f"Loading model ({MODEL_SAVE_PATH}) and tokenizer ({TOKENIZER_PATH})...")
    caption_model = load_model(MODEL_SAVE_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # 4. Evaluation & BLEU Scoring
    print("\n--- Step 4: Evaluating Test Images with BLEU Scores ---")
    if os.path.exists(FEATURES_CACHE_PATH):
        print(f"Loading cached image features from {FEATURES_CACHE_PATH}...")
        with open(FEATURES_CACHE_PATH, "rb") as f:
            cached_feats = pickle.load(f)
        test_features = cached_feats.get("test", {})
    else:
        print("Extracting features using InceptionV3...")
        cnn_model = build_inception_extractor()
        _, _, test_features = extract_and_split_features(
            model=cnn_model,
            images_directory=IMAGES_DIRECTORY,
            all_image_ids=dataset["all_image_ids"],
            train_image_ids=dataset["train_image_ids"],
            val_image_ids=dataset["val_image_ids"],
            test_image_ids=dataset["test_image_ids"],
            cache_path=FEATURES_CACHE_PATH
        )

    # Calculate greedy hypotheses for test images with available features
    eval_image_ids = [img_id for img_id in test_features if img_id in dataset["test_actual_captions"]]
    if not eval_image_ids and test_features:
        eval_image_ids = list(test_features.keys())[:5]

    greedy_caps = {}
    for img_id in eval_image_ids:
        greedy_caps[img_id] = greedy_generator(caption_model, test_features[img_id], tokenizer)

    def beam_runner(feat):
        return beam_search_generator(caption_model, feat, tokenizer, K_beams=beam_size)

    num_eval_samples = min(3, len(eval_image_ids))
    if num_eval_samples > 0:
        visualization(
            dataset["test_actual_captions"],
            test_features,
            greedy_caps,
            beam_runner,
            BLEU_score,
            num_eval_samples,
            IMAGES_DIRECTORY,
            save_path="evaluation_results.png"
        )

    # 5. Inference demonstration on sample image
    print("\n--- Step 5: Demonstration Inference ---")
    cnn_model = build_inception_extractor()
    test_img = image_path
    if not test_img or not os.path.exists(test_img):
        test_img = os.path.join(IMAGES_DIRECTORY, dataset["all_image_ids"][0])

    print(f"Inference Image: {test_img}")
    results = caption_image_file(test_img, cnn_model, caption_model, tokenizer, dataset["max_caption_length"], K_beams=beam_size)
    print(f"* Greedy Caption:   \"{results['greedy']}\"")
    print(f"* Beam Search (K={beam_size}): \"{results['beam']}\"")

    print("\n" + "=" * 65)
    print("[SUCCESS] Pipeline Completed Successfully with No Input Required!")
    print("* All artifacts generated: loss_history.png, captions_histogram.png, eda_samples.png, evaluation_results.png")
    print("* Launch interactive UI anytime: streamlit run app.py (or python app.py)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Image Caption Generator on Flickr8k")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "all", "train", "eval", "infer", "explore"],
                        help="Mode: 'auto' (default: full pipeline with zero input), 'train', 'eval', 'infer', or 'explore'")
    parser.add_argument("--image", type=str, default=None, help="Path to image for inference")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam width for beam search")
    args = parser.parse_args()

    if args.mode in ["auto", "all"]:
        run_pipeline(beam_size=args.beam_size, image_path=args.image)

    elif args.mode == "train":
        train()

    elif args.mode == "explore":
        print("Loading dataset for exploratory visualization...")
        dataset = prepare_dataset(CAPTIONS_PATH, IMAGES_DIRECTORY)
        print("Displaying caption length distribution...")
        captions_length(dataset["cleaned_captions"], save_path="captions_histogram.png")
        print("Displaying sample images with ground truth captions...")
        visualaization(dataset["captions_ids"], 5, IMAGES_DIRECTORY, save_path="eda_samples.png")
        print("Exploration plots saved to disk.")

    elif args.mode == "eval":
        if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(TOKENIZER_PATH):
            print(f"Error: Trained model ({MODEL_SAVE_PATH}) or Tokenizer ({TOKENIZER_PATH}) not found.")
            print("Please run with --mode train first.")
            return

        print("Loading model and tokenizer...")
        caption_model = load_model(MODEL_SAVE_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        dataset = prepare_dataset(CAPTIONS_PATH, IMAGES_DIRECTORY)
        cnn_model = build_inception_extractor()
        train_feats, val_feats, test_features = extract_and_split_features(
            model=cnn_model,
            images_directory=IMAGES_DIRECTORY,
            all_image_ids=dataset["all_image_ids"],
            train_image_ids=dataset["train_image_ids"],
            val_image_ids=dataset["val_image_ids"],
            test_image_ids=dataset["test_image_ids"],
            cache_path=FEATURES_CACHE_PATH
        )

        print("Generating test predictions...")
        greedy_caps = {}
        for img_id in test_features:
            greedy_caps[img_id] = greedy_generator(caption_model, test_features[img_id], tokenizer)

        def beam_runner(feat):
            return beam_search_generator(caption_model, feat, tokenizer, K_beams=args.beam_size)

        print("Rendering test visualization with BLEU scores...")
        num_eval_samples = min(5, len(dataset["test_actual_captions"]))
        visualization(dataset["test_actual_captions"], test_features, greedy_caps, beam_runner, BLEU_score, num_eval_samples, IMAGES_DIRECTORY, save_path="evaluation_results.png")

    elif args.mode == "infer":
        if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(TOKENIZER_PATH):
            print(f"Error: Trained model or Tokenizer not found. Please train the model first.")
            return

        dataset = prepare_dataset(CAPTIONS_PATH, IMAGES_DIRECTORY)
        image_path = args.image or (os.path.join(IMAGES_DIRECTORY, dataset["all_image_ids"][0]) if dataset["all_image_ids"] else None)

        if not image_path or not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return

        print("Loading models for inference...")
        cnn_model = build_inception_extractor()
        caption_model = load_model(MODEL_SAVE_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        results = caption_image_file(
            image_path,
            cnn_model,
            caption_model,
            tokenizer,
            dataset["max_caption_length"],
            K_beams=args.beam_size
        )
        print(f"\n==========================================")
        print(f"Image:            {image_path}")
        print(f"Greedy Caption:   {results['greedy']}")
        print(f"Beam Search (K={args.beam_size}): {results['beam']}")
        print(f"==========================================\n")


if __name__ == "__main__":
    main()
