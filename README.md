# 🖼️ Image Caption Generator using Deep Learning on Flickr8k Dataset

An end-to-end implementation of the **CNN (InceptionV3) + LSTM + Beam Search** image caption generator on the Flickr8k dataset, faithfully generated from the reference Google Colab / Kaggle source notebook with built-in bug fixes and performance optimizations.

---

## 📂 Project Architecture & File Structure

```text
dl/
├── Image_Caption_Generator_Flickr8k.ipynb  # Complete, self-contained Google Colab notebook
├── config.py                               # Paths, hyperparameters, dimensions
├── data_preprocessing.py                   # Caption loading, cleaning, tokenization, data generator
├── extract_features.py                     # InceptionV3 feature extraction & caching
├── model.py                                # Architecture with BatchNormalization, LSTM & Dense
├── train.py                                # Training loop, EarlyStopping, LearningRateScheduler, loss plot
├── inference.py                            # Greedy generator and Beam Search generator
├── evaluate.py                             # Exploratory EDA, BLEU evaluation, visual plotting
├── main.py                                 # Unified CLI entry point for training, evaluation, inference
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## ⚡ Key Architecture & Parameters (From Source Code)

* **Encoder (CNN):** `InceptionV3` (ImageNet pre-trained), input shape `(299, 299, 3)`, penultimate layer output (`2048` dimensions).
* **Decoder (LSTM):**
  * Image branch: `BatchNormalization` $\to$ `Dense(256, relu)` $\to$ `BatchNormalization`
  * Text branch: `Embedding(vocab_size, 256, mask_zero=True)` $\to$ `LSTM(256)`
  * Merge: `add([Image_Dense, LSTM_Out])` $\to$ `Dense(256, relu)` $\to$ `Dense(vocab_size, softmax)`
* **Training Settings:**
  * Optimizer: `Adam(learning_rate=0.01, clipnorm=1.0)`
  * Loss: `categorical_crossentropy`
  * Dynamic LR decay: `lr_scheduler(epoch, lr) = lr * exp(-0.6)`
  * Callbacks: `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`
  * Train Batch Size: `270`, Validation Batch Size: `150`
* **Inference Engines:**
  * **Greedy Search:** Fast, single-token max probability prediction.
  * **Beam Search:** Explores top $K=3$ probability trajectories with log-probability accumulation.
* **Evaluation:** BLEU-1 and BLEU-2 scoring against reference captions.

---

## 🚀 Running Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload [`Image_Caption_Generator_Flickr8k.ipynb`](./Image_Caption_Generator_Flickr8k.ipynb).
3. Switch hardware accelerator to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Click **Runtime > Run all** (`Ctrl + F9`). The notebook automatically handles dataset download, feature caching, training, and BLEU visualization!

---

## 💻 Running Option 2: Modular CLI Pipeline

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup:
Place your images and captions in `./data/`:
* `data/Images/*.jpg`
* `data/captions.txt`

*(If running on Kaggle, the script automatically picks up `/kaggle/input/flickr8k/` without any changes).*

### 3. Usage Commands:

* **Explore dataset (EDA & length distribution):**
  ```bash
  python main.py --mode explore
  ```

* **Train the model:**
  ```bash
  python main.py --mode train
  ```

* **Evaluate BLEU scores on test images:**
  ```bash
  python main.py --mode eval --beam_size 3
  ```

* **Generate caption for any custom image:**
  ```bash
  python main.py --mode infer --image path/to/your/image.jpg --beam_size 3
  ```

* **Launch the Interactive Streamlit Web UI:**
  ```bash
  streamlit run app.py
  # or directly:
  python app.py
  ```

---

## 🛠️ Built-in Bug Fixes Over Original Code
1. **Rogue token syntax error fixed:** Fixed the stray `13` on line 14 of `greedy_generator` (`replace(' end', '')13`).
2. **Missing test reference dictionary fixed:** Added `test_actual_captions` mapping needed by `visualization()`.
3. **Beam search function completion:** Reconnected the beam search routine that was severed across pages 8 and 9.
4. **Colab / Kaggle path independence:** Added automatic fallback to `kagglehub` if native Kaggle directory paths are absent.
5. **Disk feature caching:** Features are saved to `inception_features.pkl` to prevent re-extracting for 20 minutes if training is interrupted.
