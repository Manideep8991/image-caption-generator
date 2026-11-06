import os
import math
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from config import (
    IMAGES_DIRECTORY,
    CAPTIONS_PATH,
    FEATURES_CACHE_PATH,
    TOKENIZER_PATH,
    MODEL_SAVE_PATH,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_VAL,
    EPOCHS,
    LEARNING_RATE,
    CLIP_NORM,
    CNN_OUTPUT_DIM
)
from data_preprocessing import prepare_dataset, data_generator
from extract_features import build_inception_extractor, extract_and_split_features
from model import build_model, compile_caption_model


def lr_scheduler(epoch, lr):
    """Exponential learning rate decay as specified in reference implementation."""
    return float(lr * math.exp(-0.6))


def train():
    print("=== Step 1: Preprocessing Captions and Splitting Dataset ===")
    dataset = prepare_dataset(CAPTIONS_PATH, IMAGES_DIRECTORY)
    tokenizer = dataset["tokenizer"]
    vocab_size = dataset["vocab_size"]
    max_caption_length = dataset["max_caption_length"]

    print(f"Vocab size: {vocab_size}")
    print(f"Max caption length: {max_caption_length}")
    print(f"Train captions: {len(dataset['train_captions'])}")
    print(f"Val captions: {len(dataset['val_captions'])}")
    print(f"Test captions: {len(dataset['test_captions'])}")

    # Save tokenizer for inference
    print(f"Saving tokenizer to {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print("\n=== Step 2: Extracting / Loading InceptionV3 Features ===")
    cnn_model = build_inception_extractor()
    train_features, val_features, test_features = extract_and_split_features(
        model=cnn_model,
        images_directory=IMAGES_DIRECTORY,
        all_image_ids=dataset["all_image_ids"],
        train_image_ids=dataset["train_image_ids"],
        val_image_ids=dataset["val_image_ids"],
        test_image_ids=dataset["test_image_ids"],
        cache_path=FEATURES_CACHE_PATH
    )

    print("\n=== Step 3: Setting Up Data Generators ===")
    actual_batch_train = max(1, min(BATCH_SIZE_TRAIN, len(dataset["train_captions"])))
    actual_batch_val = max(1, min(BATCH_SIZE_VAL, len(dataset["val_captions"])))

    train_gen = data_generator(
        captions=dataset["train_captions"],
        image_features=train_features,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        batch_size=actual_batch_train,
        vocab_size=vocab_size
    )

    val_gen = data_generator(
        captions=dataset["val_captions"],
        image_features=val_features,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        batch_size=actual_batch_val,
        vocab_size=vocab_size
    )

    print("\n=== Step 4: Building and Compiling Caption Model ===")
    caption_model = build_model(vocab_size, max_caption_length, CNN_OUTPUT_DIM)
    caption_model = compile_caption_model(caption_model, learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
    caption_model.summary()

    print("\n=== Step 5: Training Model ===")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_schedule = LearningRateScheduler(lr_scheduler)

    steps_per_epoch = max(1, len(dataset["train_captions"]) // actual_batch_train)
    validation_steps = max(1, len(dataset["val_captions"]) // actual_batch_val)

    history = caption_model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=[early_stopping, lr_schedule]
    )

    print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
    caption_model.save(MODEL_SAVE_PATH)

    # Plotting Train VS Validation Loss
    if 'loss' in history.history:
        print("Generating loss curve plot...")
        plt.figure(figsize=(10, 5), dpi=150)
        sns.set_style('whitegrid')
        epochs_range = list(range(1, len(history.history['loss']) + 1))
        plt.plot(epochs_range, history.history['loss'], color='#E74C3C', marker='o', label='Train Loss')
        if 'val_loss' in history.history:
            plt.plot(epochs_range, history.history['val_loss'], color='#641E16', marker='h', label='Validation Loss')
        plt.title('Train VS Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Loss', fontweight='bold')
        plt.legend(loc='best')
        plot_file = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "loss_history.png")
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to {plot_file}")

    print("Training finished successfully!")


if __name__ == "__main__":
    train()
