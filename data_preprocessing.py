import os
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_captions(file_path):
    """Load captions file and convert to lowercase, skipping CSV header if present."""
    with open(file_path, 'r', encoding='utf-8') as f:
        captions = f.readlines()
    if not captions:
        return []
    # Skip header row if it contains column labels
    first_line = captions[0].strip().lower()
    if first_line.startswith(('image,caption', 'image_id,caption', 'image\tcaption')):
        captions = captions[1:]
    return [caption.lower() for caption in captions]


def clean_text(text):
    """Remove punctuation, numbers, and excess whitespace."""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_captions(captions):
    """Fit Keras Tokenizer on cleaned captions."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer


def prepare_dataset(captions_path, images_directory, test_size_1=0.15, test_size_2=0.10, random_state=42):
    """
    Parses captions, formats with start/end tokens, fits tokenizer,
    and partitions dataset into train, validation, and test sets.
    """
    raw_captions = load_captions(captions_path)
    if not raw_captions:
        raise ValueError(f"No captions found in {captions_path}")

    # Safely split on the first comma or tab so captions containing commas remain intact
    cleaned_captions = []
    img_ids = []
    for cap in raw_captions:
        cap_str = cap.strip()
        if not cap_str:
            continue
        if ',' in cap_str:
            parts = cap_str.split(',', 1)
        elif '\t' in cap_str:
            parts = cap_str.split('\t', 1)
        else:
            continue
        img_id = parts[0].strip()
        raw_text = parts[1].strip()
        cleaned = clean_text(raw_text)
        if cleaned:
            img_ids.append(img_id)
            cleaned_captions.append(cleaned)

    # Format captions as 'image_id\tstart caption end\n'
    captions_ids = []
    formatted_captions_for_tokenizer = []
    for i in range(len(cleaned_captions)):
        formatted_cap = f"start {cleaned_captions[i]} end"
        captions_ids.append(f"{img_ids[i]}\t{formatted_cap}\n")
        formatted_captions_for_tokenizer.append(formatted_cap)

    # Tokenizer & vocabulary sizing (fitted on formatted captions to guarantee 'start' & 'end' are in vocab)
    tokenizer = tokenize_captions(formatted_captions_for_tokenizer)
    vocab_size = len(tokenizer.word_index) + 1
    max_caption_length = max(len(cap.split()) for cap in formatted_captions_for_tokenizer)

    # Filter to image files only
    all_image_ids = sorted([
        f for f in os.listdir(images_directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not all_image_ids:
        raise ValueError(f"No image files (.jpg, .png) found in {images_directory}")

    # Robust Train / Val / Test splitting
    num_images = len(all_image_ids)
    if num_images >= 20:
        train_image_ids, val_image_ids = train_test_split(all_image_ids, test_size=test_size_1, random_state=random_state)
        val_image_ids, test_image_ids = train_test_split(val_image_ids, test_size=test_size_2, random_state=random_state)
    else:
        # Fallback for small sample/starter datasets
        n_train = max(1, int(num_images * 0.7))
        n_val = max(1, int(num_images * 0.15))
        train_image_ids = all_image_ids[:n_train]
        val_image_ids = all_image_ids[n_train:n_train + n_val]
        test_image_ids = all_image_ids[n_train + n_val:]
        if not test_image_ids:
            test_image_ids = val_image_ids

    train_captions, val_captions, test_captions = [], [], []
    for caption in captions_ids:
        image_id, _ = caption.split('\t', 1)
        if image_id in train_image_ids:
            train_captions.append(caption)
        elif image_id in val_image_ids:
            val_captions.append(caption)
        elif image_id in test_image_ids:
            test_captions.append(caption)

    # Build reference mapping for test evaluation
    test_actual_captions = {}
    for item in test_captions:
        img_id, cap = item.split('\t', 1)
        if img_id not in test_actual_captions:
            test_actual_captions[img_id] = []
        test_actual_captions[img_id].append(cap.strip())

    return {
        "raw_captions": raw_captions,
        "cleaned_captions": cleaned_captions,
        "captions_ids": captions_ids,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "max_caption_length": max_caption_length,
        "all_image_ids": all_image_ids,
        "train_image_ids": train_image_ids,
        "val_image_ids": val_image_ids,
        "test_image_ids": test_image_ids,
        "train_captions": train_captions,
        "val_captions": val_captions,
        "test_captions": test_captions,
        "test_actual_captions": test_actual_captions
    }


def data_generator(captions, image_features, tokenizer, max_caption_length, batch_size, vocab_size):
    """
    Generator yielding batches of (X_images, X_captions) and y one-hot encoded targets.
    """
    # Pre-filter captions that have corresponding extracted features
    valid_captions = [c for c in captions if c.split('\t')[0] in image_features]
    if not valid_captions:
        # Fallback to avoid empty generator
        valid_captions = captions

    num_samples = len(valid_captions)
    actual_batch_size = max(1, min(batch_size, num_samples))

    while True:
        # Shuffle captions for each epoch
        indices = np.random.permutation(num_samples)
        shuffled_captions = [valid_captions[i] for i in indices]

        for start_idx in range(0, num_samples, actual_batch_size):
            end_idx = min(start_idx + actual_batch_size, num_samples)
            X_images, X_captions, y = [], [], []

            for caption in shuffled_captions[start_idx:end_idx]:
                image_id, caption_text = caption.split('\t', 1)
                caption_text = caption_text.rstrip('\n')
                if image_id not in image_features:
                    continue

                seq = tokenizer.texts_to_sequences([caption_text])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_caption_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    X_images.append(image_features[image_id])
                    X_captions.append(in_seq)
                    y.append(out_seq)

            if len(X_images) > 0:
                yield (np.array(X_images, dtype=np.float32), np.array(X_captions, dtype=np.int32)), np.array(y, dtype=np.float32)


if __name__ == "__main__":
    from config import CAPTIONS_PATH, IMAGES_DIRECTORY
    print("=== Testing Data Preprocessing Pipeline ===")
    dataset = prepare_dataset(CAPTIONS_PATH, IMAGES_DIRECTORY)
    print(f"Loaded raw captions:     {len(dataset['raw_captions'])}")
    print(f"Vocabulary size:         {dataset['vocab_size']}")
    print(f"Max caption length:      {dataset['max_caption_length']}")
    print(f"Images in dataset:       {len(dataset['all_image_ids'])}")
    print(f"Train / Val / Test imgs: {len(dataset['train_image_ids'])} / {len(dataset['val_image_ids'])} / {len(dataset['test_image_ids'])}")
    print(f"Train / Val / Test caps: {len(dataset['train_captions'])} / {len(dataset['val_captions'])} / {len(dataset['test_captions'])}")
    print(f"Start token index:       {dataset['tokenizer'].word_index.get('start')}")
    print(f"End token index:         {dataset['tokenizer'].word_index.get('end')}")
    print("Data preprocessing passed successfully!")

