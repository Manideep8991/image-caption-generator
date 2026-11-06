import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def visualaization(data, num_of_images, images_directory, save_path=None):
    """
    Plots sample images along with their ground-truth captions.
    Faithful to Page 2 of the reference notebook.
    """
    captions_dictionary = {}
    slice_data = data[:min(len(data), (num_of_images) * 5)]
    for item in slice_data:
        image_id, caption = item.split('\t', 1)
        if image_id not in captions_dictionary:
            captions_dictionary[image_id] = []
        captions_dictionary[image_id].append(caption)

    num_images_to_plot = min(num_of_images, len(captions_dictionary))
    if num_images_to_plot == 0:
        print("No images found to visualize.")
        return

    count = 1
    fig = plt.figure(figsize=(10, num_images_to_plot * 4))
    for filename in list(captions_dictionary.keys())[:num_images_to_plot]:
        captions = captions_dictionary[filename]
        image_path = os.path.join(images_directory, filename.strip())
        if not os.path.exists(image_path):
            continue
        image_load = load_img(image_path, target_size=(199, 199))
        ax = fig.add_subplot(num_images_to_plot, 2, count, xticks=[], yticks=[])
        ax.imshow(image_load)
        count += 1

        ax = fig.add_subplot(num_images_to_plot, 2, count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(captions) + 1)
        for i, caption in enumerate(captions):
            clean_display = caption.replace("start", "").replace("end", "").strip()
            ax.text(0, i + 0.5, f"• {clean_display}", fontsize=11)
        count += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Sample visualization saved to {save_path}")
    else:
        try:
            plt.show(block=False)
            plt.pause(1)
        except Exception:
            pass
    plt.close()


# Backward compatible alias for typo
visualize_samples = visualaization


def captions_length(data, save_path=None):
    """Plots histogram of caption word counts. Faithful to Page 3."""
    plt.figure(figsize=(12, 6), dpi=150)
    sns.set_style('darkgrid')
    lengths = [len(x.split()) for x in data]
    sns.histplot(x=lengths, kde=True, discrete=True)
    plt.title('Captions length histogram', fontsize=14, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xlabel('Length (words)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Length histogram saved to {save_path}")
    else:
        try:
            plt.show(block=False)
            plt.pause(1)
        except Exception:
            pass
    plt.close()


def BLEU_score(actual, greedy, beam_search):
    """
    Computes BLEU-1 and BLEU-2 scores for greedy and beam search hypotheses
    against reference captions. Faithful to Page 9 of the notebook.
    """
    # Smooth to avoid zero scores for short sentences
    smooth = SmoothingFunction().method1

    # Format actual as list of tokenized references: [[token1, token2, ...], ...]
    clean_actual = [ref.replace("start", "").replace("end", "").strip() for ref in actual]
    tokenized_actual = [[ref.split() for ref in clean_actual]]

    greedy_text = greedy[0] if isinstance(greedy, (list, tuple)) and greedy else str(greedy)
    beam_text = beam_search[0] if isinstance(beam_search, (list, tuple)) and beam_search else str(beam_search)

    tokenized_greedy = [greedy_text.split()]
    tokenized_beam = [beam_text.split()]

    score_greedy_1 = corpus_bleu(tokenized_actual, tokenized_greedy, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    score_greedy_2 = corpus_bleu(tokenized_actual, tokenized_greedy, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    score_BS_1 = corpus_bleu(tokenized_actual, tokenized_beam, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    score_BS_2 = corpus_bleu(tokenized_actual, tokenized_beam, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)

    return [
        f'BLEU-1 Greedy: {round(score_greedy_1, 4)}',
        f'BLEU-2 Greedy: {round(score_greedy_2, 4)}',
        f'Greedy: {greedy_text}',
        f'BLEU-1 Beam Search: {round(score_BS_1, 4)}',
        f'BLEU-2 Beam Search: {round(score_BS_2, 4)}',
        f'Beam Search: {beam_text}'
    ]


def visualization(data, test_image_features, greedy_caps, beamS_generator, evaluator, num_of_images, images_directory, save_path=None):
    """
    Displays test images along with reference captions, greedy prediction,
    beam search prediction, and computed BLEU scores. Faithful to Page 9 & 10.
    """
    keys = list(data.keys())
    if not keys:
        print("No test data found for visualization.")
        return

    num_samples = min(num_of_images, len(keys))
    selected_images = list(np.random.choice(keys, size=num_samples, replace=False))
    count = 1
    fig = plt.figure(figsize=(12, num_samples * 5))

    for filename in selected_images:
        actual_cap = data[filename]
        actual_cap = [x.replace("start ", "").replace("start", "").strip() for x in actual_cap]
        actual_cap = [x.replace(" end", "").replace("end", "").strip() for x in actual_cap]

        greedy_cap = greedy_caps.get(filename, "")
        feat = test_image_features.get(filename)
        beamS_cap = beamS_generator(feat) if feat is not None else ""

        caps_with_score = evaluator(
            actual_cap,
            [greedy_cap],
            [beamS_cap]
        )

        image_path = os.path.join(images_directory, filename.strip())
        if os.path.exists(image_path):
            image_load = load_img(image_path, target_size=(199, 199))
            ax = fig.add_subplot(num_samples, 2, count, xticks=[], yticks=[])
            ax.imshow(image_load)
        else:
            ax = fig.add_subplot(num_samples, 2, count)
            ax.text(0.5, 0.5, f"Image not found:\n{filename}", ha='center')
        count += 1

        ax = fig.add_subplot(num_samples, 2, count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(caps_with_score) + 1)
        for i, text in enumerate(reversed(caps_with_score)):
            ax.text(0, i + 0.5, text, fontsize=10)
        count += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Evaluation visualization saved to {save_path}")
    else:
        try:
            plt.show(block=False)
            plt.pause(1)
        except Exception:
            pass
    plt.close()


if __name__ == "__main__":
    from config import CAPTIONS_PATH, IMAGES_DIRECTORY
    from data_preprocessing import prepare_dataset

    print("=== Testing Evaluation Module ===")
    dataset = prepare_dataset(CAPTIONS_PATH, IMAGES_DIRECTORY)
    print("Testing visualaization function...")
    visualaization(dataset["captions_ids"], 3, IMAGES_DIRECTORY, save_path="eda_samples.png")
    print("Testing captions_length function...")
    captions_length(dataset["cleaned_captions"], save_path="captions_histogram.png")
    print("Evaluation module test passed!")

