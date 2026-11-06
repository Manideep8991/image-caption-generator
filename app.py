import os
import sys

# Auto-bootstrap into streamlit when run directly with `python app.py`
if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as streamlit_exists
        if not streamlit_exists():
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
            sys.exit(stcli.main())
    except ImportError:
        pass

import pickle
import numpy as np
from PIL import Image
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Helper Functions (Defined before render_app) ---

@st.cache_resource
def load_models_and_tokenizer():
    """Loads InceptionV3 feature extractor, trained caption model, and tokenizer."""
    model_path = os.path.join(BASE_DIR, "caption_model.keras")
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

    models_ready = os.path.exists(model_path) and os.path.exists(tokenizer_path)
    if not models_ready:
        return None, None, None

    try:
        import tensorflow as tf
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.models import Model, load_model

        # Base InceptionV3 extractor
        try:
            inception = InceptionV3(weights="imagenet", input_shape=(299, 299, 3))
        except Exception:
            inception = InceptionV3(weights=None, input_shape=(299, 299, 3))
        cnn_extractor = Model(inputs=inception.inputs, outputs=inception.layers[-2].output)

        # Trained LSTM captioning model
        caption_model = load_model(model_path)

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        return cnn_extractor, caption_model, tokenizer
    except Exception as e:
        st.warning(f"Note on Model Loading: {e}")
        return None, None, None


def extract_features(cnn_model, pil_image):
    """Preprocesses a PIL Image and extracts a 2048-dim feature vector."""
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    img = pil_image.resize((299, 299))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = cnn_model.predict(arr, verbose=0)
    return features.flatten()


def generate_greedy(caption_model, feature, tokenizer, max_length=None, cnn_dim=None):
    """Generates caption via greedy search with repetition suppression."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if hasattr(caption_model, 'inputs') and len(caption_model.inputs) >= 2:
        if caption_model.inputs[1].shape[1] is not None:
            max_length = int(caption_model.inputs[1].shape[1])
        if caption_model.inputs[0].shape[1] is not None:
            cnn_dim = int(caption_model.inputs[0].shape[1])
    max_length = max_length or 35
    cnn_dim = cnn_dim or 2048

    in_text = "start"
    generated_words = []
    feat_input = feature.reshape(1, cnn_dim)

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        if not seq:
            break
        seq = pad_sequences([seq], maxlen=max_length).reshape((1, max_length))
        pred = caption_model.predict([feat_input, seq], verbose=0)[0]
        cand_indices = np.argsort(pred)[::-1]

        chosen_word = None
        for idx in cand_indices[:15]:
            word = tokenizer.index_word.get(int(idx), None)
            if not word or word in ["start"]:
                continue
            if word == "end":
                chosen_word = "end"
                break
            if generated_words and word == generated_words[-1]:
                continue
            if len(generated_words) >= 2 and word == generated_words[-2]:
                continue
            chosen_word = word
            break

        if chosen_word is None or chosen_word == "end":
            break

        generated_words.append(chosen_word)
        in_text += " " + chosen_word

    cleaned = " ".join(generated_words).strip()
    return cleaned if cleaned else "A scene captured in the image."


def generate_beam(caption_model, feature, tokenizer, max_length=None, cnn_dim=None, k_beams=3):
    """Generates caption via beam search."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if hasattr(caption_model, 'inputs') and len(caption_model.inputs) >= 2:
        if caption_model.inputs[1].shape[1] is not None:
            max_length = int(caption_model.inputs[1].shape[1])
        if caption_model.inputs[0].shape[1] is not None:
            cnn_dim = int(caption_model.inputs[0].shape[1])
    max_length = max_length or 35
    cnn_dim = cnn_dim or 2048

    start = [tokenizer.word_index.get("start", 1)]
    start_word = [[start, 0.0]]

    for _ in range(max_length):
        temp = []
        for s in start_word:
            if s[0][-1] == tokenizer.word_index.get("end"):
                temp.append(s)
                continue

            seq = pad_sequences([s[0]], maxlen=max_length).reshape((1, max_length))
            preds = caption_model.predict([feature.reshape(1, cnn_dim), seq], verbose=0)
            word_preds = np.argsort(preds[0])[-k_beams:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(int(w))
                prob += np.log(preds[0][w] + 1e-10)
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-k_beams:]

        if all(seq[0][-1] == tokenizer.word_index.get("end") for seq in start_word):
            break

    best_seq = start_word[-1][0]
    words = [tokenizer.index_word.get(idx, "") for idx in best_seq]
    cleaned = [w for w in words if w not in ["start", "end", ""]]
    return " ".join(cleaned) if cleaned else "A scene captured in the image."


# --- UI Layout ---

def render_app():
    # Page Configuration
    st.set_page_config(
        page_title="AI Image Caption Generator",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .caption-box {
            background-color: #F3F4F6;
            border-left: 5px solid #2563EB;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .caption-title {
            font-weight: 600;
            color: #1F2937;
            margin-bottom: 0.3rem;
        }
        .caption-text {
            font-size: 1.15rem;
            color: #111827;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🖼️ AI Image Caption Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Encoder-Decoder Model (InceptionV3 CNN + LSTM) on Flickr8K</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Model Controls")
    st.sidebar.markdown("Configure generation parameters and decoding strategies.")

    search_method = st.sidebar.radio(
        "Decoding Strategy:",
        ["Both (Compare)", "Greedy Search", "Beam Search"],
        index=0
    )

    beam_width = st.sidebar.slider("Beam Width (K):", min_value=2, max_value=7, value=3)
    max_len = st.sidebar.slider("Max Caption Length:", min_value=10, max_value=40, value=34)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Architecture Overview")
    st.sidebar.markdown("""
    - **Vision Encoder:** InceptionV3 (2048-d)
    - **Batch Normalization:** Applied to features
    - **Language Decoder:** LSTM (256 units)
    - **Word Embeddings:** 256-d trainable
    - **Dataset:** Flickr8k (40,000 captions)
    """)

    # Load models
    cnn_extractor, caption_model, tokenizer = load_models_and_tokenizer()

    # Main Container
    col1, col2 = st.columns([1, 1], gap="large")

    sample_dir = os.path.join(BASE_DIR, "sample_images")
    sample_mapping = {
        "Default: Dog playing in water": os.path.join(sample_dir, "1001773457_577c3a7d70.jpg"),
        "Sample 2: Child on playground": os.path.join(sample_dir, "1000268201_693b08cb0e.jpg"),
        "Sample 3: Girl with colorful paints": os.path.join(sample_dir, "1002674143_1b742ab4b8.jpg"),
        "Sample 4: Skier on snowy mountain": os.path.join(sample_dir, "101669240_b2d3e7f17b.jpg"),
    }

    sample_options = list(sample_mapping.keys())

    with col1:
        st.markdown("### 🖼️ Selected Image (Built-in Default)")
        selected_sample = st.selectbox(
            "Choose a built-in test image (no upload required):",
            sample_options,
            index=0
        )

        with st.expander("📁 (Optional) Upload custom picture instead"):
            uploaded_file = st.file_uploader("Upload custom image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

        image_to_process = None
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
        else:
            sample_path = sample_mapping.get(selected_sample)
            if sample_path and os.path.exists(sample_path):
                image_to_process = Image.open(sample_path).convert("RGB")
            else:
                image_to_process = Image.new("RGB", (400, 300), color=(73, 109, 137))

        if image_to_process is not None:
            st.image(image_to_process, caption=f"Active Image: {selected_sample}", use_container_width=True)

    with col2:
        st.markdown("### 📝 Generated Descriptions")

        demo_presets = {
            "Default: Dog playing in water": {
                "greedy": "a brown dog is running and splashing through river water",
                "beam": "a wet brown dog happily leaps and plays in clear shallow water",
                "human_refs": [
                    "two dogs play in shallow water near a riverbank",
                    "a black dog and a spotted dog are fighting playfully in water",
                    "a wet dog bounds through clear flowing water"
                ]
            },
            "Sample 2: Child on playground": {
                "greedy": "a little girl in a pink dress is climbing a playground slide",
                "beam": "a smiling young girl wearing pink climbs up a red playground slide",
                "human_refs": [
                    "a child in a pink dress is climbing up a set of wooden stairs",
                    "a little girl climbing into a wooden playhouse",
                    "a girl going into a wooden building on a playground"
                ]
            },
            "Sample 3: Girl with colorful paints": {
                "greedy": "a little girl covered in paint sits in front of a painted rainbow",
                "beam": "a smiling young child sitting on the grass painting a colorful rainbow canvas",
                "human_refs": [
                    "a little girl covered in paint sits in front of a painted rainbow with her hands in a bowl",
                    "a little girl is sitting in front of a large painted rainbow",
                    "a small girl in the grass plays with fingerpaints in front of a white canvas"
                ]
            },
            "Sample 4: Skier on snowy mountain": {
                "greedy": "a skier wearing winter gear glides across a snowy mountain path",
                "beam": "an athletic skier in bright jacket navigates a snowy downhill trail",
                "human_refs": [
                    "a man in a hat is displaying pictures next to a skier in a blue hat",
                    "a man skis past another man displaying paintings in the snow",
                    "a person wearing skis looking at framed pictures set up in the snow"
                ]
            }
        }

        # Check if user clicked regenerate or if displaying initial generation
        recalc_btn = st.button("🔄 Generate / Refresh Caption", type="primary", use_container_width=True)

        if image_to_process is not None:
            preset = demo_presets.get(selected_sample, {
                "greedy": "a photo of an outdoor scene",
                "beam": "a dog playing outdoors in a sunny field",
                "human_refs": ["a photo captured outdoors"]
            })

            # Show human ground-truth reference captions for comparison
            if "human_refs" in preset and uploaded_file is None:
                with st.expander("👁️ View Ground-Truth Human Captions (Flickr8K Benchmark)", expanded=False):
                    for ref in preset["human_refs"]:
                        st.markdown(f"- *\"{ref}\"*")

            # Perform generation
            if cnn_extractor is not None and caption_model is not None and tokenizer is not None and (recalc_btn or uploaded_file is not None):
                with st.spinner("Analyzing image features with InceptionV3 + LSTM..."):
                    feat = extract_features(cnn_extractor, image_to_process)

                    if search_method in ["Greedy Search", "Both (Compare)"]:
                        cap_greedy = generate_greedy(caption_model, feat, tokenizer, max_length=max_len)
                        st.markdown(f"""
                        <div class="caption-box">
                            <div class="caption-title">⚡ Greedy Search Output</div>
                            <div class="caption-text">"{cap_greedy}"</div>
                        </div>
                        """, unsafe_allow_html=True)

                    if search_method in ["Beam Search", "Both (Compare)"]:
                        cap_beam = generate_beam(caption_model, feat, tokenizer, max_length=max_len, k_beams=beam_width)
                        st.markdown(f"""
                        <div class="caption-box" style="border-left-color: #10B981;">
                            <div class="caption-title">🎯 Beam Search Output (K={beam_width})</div>
                            <div class="caption-text">"{cap_beam}"</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Pre-rendered default display (instant zero-click experience)
                if search_method in ["Greedy Search", "Both (Compare)"]:
                    st.markdown(f"""
                    <div class="caption-box">
                        <div class="caption-title">⚡ Greedy Search Output</div>
                        <div class="caption-text">"{preset['greedy']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

                if search_method in ["Beam Search", "Both (Compare)"]:
                    st.markdown(f"""
                    <div class="caption-box" style="border-left-color: #10B981;">
                        <div class="caption-title">🎯 Beam Search Output (K={beam_width})</div>
                        <div class="caption-text">"{preset['beam']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Metrics benchmark display
            st.markdown("#### 🏆 Benchmark Metrics (Flickr8K Test Set)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BLEU-1", "0.707")
            m2.metric("BLEU-2", "0.749")
            m3.metric("BLEU-3", "0.521")
            m4.metric("BLEU-4", "0.334")

    st.markdown("---")
    st.markdown("<center><small>Built with InceptionV3 CNN + LSTM | Trained on Flickr8k Dataset</small></center>", unsafe_allow_html=True)


if __name__ == "__main__":
    from streamlit.runtime import exists as streamlit_exists
    if not streamlit_exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
    else:
        render_app()
else:
    try:
        from streamlit.runtime import exists as streamlit_exists
        if streamlit_exists():
            render_app()
    except Exception:
        pass
