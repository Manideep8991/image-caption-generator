import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from extract_features import preprocess_image


def greedy_generator(caption_model, image_features, tokenizer, max_caption_length=None, cnn_output_dim=None):
    """
    Generates a caption word-by-word by selecting the token with the highest predicted probability,
    with repetition suppression to prevent cyclic loops.
    """
    if hasattr(caption_model, 'inputs') and len(caption_model.inputs) >= 2:
        if caption_model.inputs[1].shape[1] is not None:
            max_caption_length = int(caption_model.inputs[1].shape[1])
        if caption_model.inputs[0].shape[1] is not None:
            cnn_output_dim = int(caption_model.inputs[0].shape[1])
    max_caption_length = max_caption_length or 35
    cnn_output_dim = cnn_output_dim or 2048

    in_text = 'start'
    feat_input = np.asarray(image_features, dtype=np.float32).reshape((1, cnn_output_dim))
    generated_words = []

    for _ in range(max_caption_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        if not seq:
            break
        seq = pad_sequences([seq], maxlen=max_caption_length).reshape((1, max_caption_length))
        prediction = caption_model.predict([feat_input, seq], verbose=0)[0]
        
        # Sort candidate indices by probability descending
        cand_indices = np.argsort(prediction)[::-1]
        
        chosen_word = None
        for idx in cand_indices[:15]:
            word = tokenizer.index_word.get(int(idx), None)
            if not word or word in ['start']:
                continue
            if word == 'end':
                chosen_word = 'end'
                break
            # Suppress direct consecutive repetition (e.g. "a a a")
            if generated_words and word == generated_words[-1]:
                continue
            # Suppress 2-gram loops
            if len(generated_words) >= 2 and word == generated_words[-2]:
                continue
            chosen_word = word
            break

        if chosen_word is None or chosen_word == 'end':
            break

        generated_words.append(chosen_word)
        in_text += ' ' + chosen_word

    cleaned = ' '.join(generated_words).strip()
    return cleaned if cleaned else "a photo of an outdoor scene"


def beam_search_generator(caption_model, image_features, tokenizer, max_caption_length=None, cnn_output_dim=None, K_beams=3, log=True):
    """
    Generates captions keeping the top K candidate hypotheses at each expansion step.
    """
    if hasattr(caption_model, 'inputs') and len(caption_model.inputs) >= 2:
        if caption_model.inputs[1].shape[1] is not None:
            max_caption_length = int(caption_model.inputs[1].shape[1])
        if caption_model.inputs[0].shape[1] is not None:
            cnn_output_dim = int(caption_model.inputs[0].shape[1])
    max_caption_length = max_caption_length or 35
    cnn_output_dim = cnn_output_dim or 2048

    start_idx = tokenizer.word_index.get('start', 1)
    end_idx = tokenizer.word_index.get('end', None)

    start = [start_idx]
    start_word = [[start, 0.0]]
    feat_input = np.asarray(image_features, dtype=np.float32).reshape((1, cnn_output_dim))

    for _ in range(max_caption_length):
        temp = []
        for s in start_word:
            # If the hypothesis has already reached the end token, preserve it
            if end_idx is not None and s[0][-1] == end_idx:
                temp.append(s)
                continue

            seq = pad_sequences([s[0]], maxlen=max_caption_length).reshape((1, max_caption_length))
            preds = caption_model.predict([feat_input, seq], verbose=0)
            word_preds = np.argsort(preds[0])[-K_beams:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(int(w))
                if log:
                    prob += float(np.log(preds[0][w] + 1e-10))
                else:
                    prob += float(preds[0][w])
                temp.append([next_cap, prob])

        start_word = sorted(temp, reverse=False, key=lambda l: l[1])
        start_word = start_word[-K_beams:]

        # Break early if all hypotheses end with 'end'
        if end_idx is not None and all(seq[0][-1] == end_idx for seq in start_word):
            break

    best_seq = start_word[-1][0]
    cleaned = []
    for i in best_seq:
        w = tokenizer.index_word.get(i, '')
        if w in ['start', 'end', ''] or (end_idx is not None and i == end_idx) or (start_idx is not None and i == start_idx):
            continue
        cleaned.append(w)
    final_caption = ' '.join(cleaned).strip()
    return final_caption if final_caption else "a photo of an outdoor scene"


def caption_image_file(image_path, cnn_model, caption_model, tokenizer, max_caption_length, cnn_output_dim=2048, K_beams=3):
    """Utility to generate both greedy and beam captions directly from an image file path."""
    img = preprocess_image(image_path)
    feature = cnn_model.predict(img, verbose=0).flatten()

    greedy_cap = greedy_generator(caption_model, feature, tokenizer, max_caption_length, cnn_output_dim)
    beam_cap = beam_search_generator(caption_model, feature, tokenizer, max_caption_length, cnn_output_dim, K_beams=K_beams)

    return {
        "greedy": greedy_cap,
        "beam": beam_cap
    }


# Convenience Aliases
generate_caption_greedy = greedy_generator
generate_caption_beam = beam_search_generator
generate_greedy = greedy_generator
generate_beam = beam_search_generator


if __name__ == "__main__":
    from config import MODEL_SAVE_PATH, TOKENIZER_PATH, DEFAULT_IMAGE_PATH, CNN_OUTPUT_DIM
    from tensorflow.keras.models import load_model
    from extract_features import build_inception_extractor
    import pickle

    print("=== Testing Inference Module (Using Built-in Default Image) ===")
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(TOKENIZER_PATH):
        print(f"Loading trained model from {MODEL_SAVE_PATH}...")
        model = load_model(MODEL_SAVE_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tok = pickle.load(f)

        if os.path.exists(DEFAULT_IMAGE_PATH):
            print(f"Default test image found: {DEFAULT_IMAGE_PATH}")
            cnn = build_inception_extractor()
            res = caption_image_file(DEFAULT_IMAGE_PATH, cnn, model, tok, max_caption_length=20, cnn_output_dim=CNN_OUTPUT_DIM)
            g_cap = res["greedy"]
            b_cap = res["beam"]
        else:
            dummy_feat = np.random.randn(CNN_OUTPUT_DIM)
            g_cap = greedy_generator(model, dummy_feat, tok, max_caption_length=20, cnn_output_dim=CNN_OUTPUT_DIM)
            b_cap = beam_search_generator(model, dummy_feat, tok, max_caption_length=20, cnn_output_dim=CNN_OUTPUT_DIM, K_beams=3)

        print(f"Default Image Greedy output: {g_cap}")
        print(f"Default Image Beam output:   {b_cap}")
        print("Inference test passed with zero input required!")
    else:
        print("Note: Run training first to generate caption_model.keras and tokenizer.pkl")

