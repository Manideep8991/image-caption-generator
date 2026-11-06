import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tqdm import tqdm


def build_inception_extractor(weights='imagenet'):
    """
    Initializes InceptionV3 with ImageNet weights (or random initialization if offline),
    removes top layer, and returns model outputting penultimate layer features.
    """
    try:
        inception = InceptionV3(weights=weights, input_shape=(299, 299, 3))
    except Exception as e:
        print(f"Note: Could not fetch remote ImageNet weights ({e}). Initializing InceptionV3 architecture locally.")
        inception = InceptionV3(weights=None, input_shape=(299, 299, 3))
    # Using penultimate output layer (2048-dim representation)
    model = Model(inputs=inception.inputs, outputs=inception.layers[-2].output)
    return model


def preprocess_image(image_path):
    """Loads and preprocesses a single image for InceptionV3."""
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def extract_single_feature(model, image_path):
    """Extracts and returns flattened 2048-dim feature vector for one image."""
    img = preprocess_image(image_path)
    feature = model.predict(img, verbose=0)
    return feature.flatten()


def extract_and_split_features(model, images_directory, all_image_ids, train_image_ids, val_image_ids, test_image_ids, cache_path=None):
    """
    Extracts features for all images and partitions them into train, val, and test feature dictionaries.
    Caches results to disk if cache_path is specified.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["val"], data["test"]

    train_features, val_features, test_features = {}, {}, {}
    print(f"Extracting features for {len(all_image_ids)} images using InceptionV3...")

    for img_name in tqdm(all_image_ids, desc="Extracting features"):
        image_id = img_name.split('\t')[0]
        image_path = os.path.join(images_directory, image_id)

        if not os.path.exists(image_path):
            continue

        feat = extract_single_feature(model, image_path)

        if image_id in train_image_ids:
            train_features[image_id] = feat
        elif image_id in val_image_ids:
            val_features[image_id] = feat
        elif image_id in test_image_ids:
            test_features[image_id] = feat

    if cache_path:
        print(f"Saving extracted features to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "train": train_features,
                "val": val_features,
                "test": test_features
            }, f)

    return train_features, val_features, test_features
