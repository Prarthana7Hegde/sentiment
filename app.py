"""
Movie Review Sentiment Analysis - Streamlit interface
Model: GRU trained on 50,000 IMDB reviews (accuracy 88.8%, ROC-AUC 0.952)

Run:  streamlit run app.py
Needs in the same folder: best_sentiment_model.keras, vocab.json
"""
import io
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "best_sentiment_model.keras"
VOCAB_PATH = "vocab.json"
MAX_LEN = 256          # must match training

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")


# ----------------------------------------------------------------- loading
def build_gru_model():
    """Same architecture as the training notebook (GRU model)."""
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = layers.Input(shape=(MAX_LEN,))
    x = layers.Embedding(20000, 128)(inp)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="GRU")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    from tensorflow import keras
    try:
        return keras.models.load_model(path)
    except Exception:
        # Older Keras can't read the saved config (version mismatch).
        # Rebuild the same architecture and load only the trained weights.
        model = build_gru_model()
        model.load_weights(path)
        return model


@st.cache_resource(show_spinner=False)
def load_vocab_file(path):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------------- preprocessing
# identical to the training notebook
def clean_text(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.lower()
    t = re.sub(r"[^a-z\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def encode(texts, vocab, maxlen=MAX_LEN):
    X = np.zeros((len(texts), maxlen), dtype="int32")
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in clean_text(t).split()][:maxlen]
        if ids:
            X[i, -len(ids):] = ids          # pre-padding
    return X


def predict_proba(texts, model, vocab):
    return model.predict(encode(texts, vocab), batch_size=256, verbose=0).ravel()


def verdict(p):
    """Return (label, emoji, confidence) from the positive probability."""
    if 0.40 <= p <= 0.60:
        return "Borderline / Mixed", "😐", max(p, 1 - p)
    if p > 0.60:
        return "Positive", "😊", p
    return "Negative", "😞", 1 - p


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About")
    st.write("A deep-learning classifier that labels movie reviews as "
             "**positive** or **negative**.")
    st.markdown("**Model:** GRU  \n**Data:** IMDB 50K reviews  \n"
                f"**Input length:** {MAX_LEN} words")
    st.subheader("Test-set performance")
    st.table(pd.DataFrame(
        {"Score": [0.888, 0.873, 0.908, 0.890, 0.952]},
        index=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]))
    st.caption("JSS Science and Technology University - Deep Learning (22IS641)")

# ------------------------------------------------------------ load files
if not os.path.exists(MODEL_PATH):
    st.error(f"`{MODEL_PATH}` not found. Put it in the same folder as app.py.")
    st.stop()

if os.path.exists(VOCAB_PATH):
    vocab = load_vocab_file(VOCAB_PATH)
else:
    st.warning("`vocab.json` not found next to app.py. Upload it below "
               "(it was saved with the model in Google Drive).")
    up = st.file_uploader("Upload vocab.json", type="json")
    if up is None:
        st.stop()
    vocab = json.load(io.TextIOWrapper(up, encoding="utf-8"))

model = load_model(MODEL_PATH)

# --------------------------------------------------------------------- UI
st.title("🎬 Movie Review Sentiment Analysis")
st.caption("Type or paste a review and the GRU model predicts its sentiment.")

if "history" not in st.session_state:
    st.session_state.history = []
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV)"])

# ---------------------------------------------------------- single review
with tab_single:
    examples = {
        "Positive example": "An absolute masterpiece. Brilliant acting, a gripping story and a beautiful soundtrack.",
        "Negative example": "Boring, predictable and a complete waste of my time. The acting was awful.",
        "Mixed example": "The visuals were stunning but the plot was weak and the ending felt rushed.",
    }
    cols = st.columns(len(examples))
    for col, (name, text) in zip(cols, examples.items()):
        if col.button(name, width="stretch"):
            st.session_state.review_text = text

    review = st.text_area("Your review", key="review_text", height=170,
                          placeholder="e.g. I loved this movie, the performances were superb...")

    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            p = float(predict_proba([review], model, vocab)[0])
            label, emoji, conf = verdict(p)

            st.subheader(f"{emoji} {label}")
            st.progress(p, text=f"Positive probability: {p:.1%}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", f"{p:.1%}")
            c2.metric("Negative", f"{1 - p:.1%}")
            c3.metric("Words used", f"{min(len(clean_text(review).split()), MAX_LEN)} / {MAX_LEN}")

            if len(clean_text(review).split()) > MAX_LEN:
                st.info(f"Review is longer than {MAX_LEN} words; only the first {MAX_LEN} were analyzed.")
            if label.startswith("Borderline"):
                st.info("The model is unsure - the review probably contains mixed opinions.")

            st.session_state.history.insert(0, {
                "Review": review[:80] + ("..." if len(review) > 80 else ""),
                "Sentiment": label,
                "Positive prob.": round(p, 3),
            })

    if st.session_state.history:
        st.divider()
        st.subheader("Recent predictions")
        st.dataframe(pd.DataFrame(st.session_state.history[:10]), width="stretch", hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Upload a CSV with a column of reviews to classify them all at once.")
    csv = st.file_uploader("CSV file", type="csv", key="csv_up")
    if csv is not None:
        data = pd.read_csv(csv)
        st.dataframe(data.head(), width="stretch")
        col = st.selectbox("Column containing the reviews", data.columns)
        if st.button("Run batch prediction"):
            texts = data[col].astype(str).tolist()
            with st.spinner(f"Classifying {len(texts)} reviews..."):
                probs = predict_proba(texts, model, vocab)
            out = data.copy()
            out["positive_probability"] = probs.round(4)
            out["predicted_sentiment"] = np.where(probs > 0.5, "Positive", "Negative")

            n_pos = int((probs > 0.5).sum())
            m1, m2 = st.columns(2)
            m1.metric("Positive", n_pos)
            m2.metric("Negative", len(probs) - n_pos)
            st.bar_chart(out["predicted_sentiment"].value_counts())
            st.dataframe(out, width="stretch")
            st.download_button("Download results (CSV)",
                               out.to_csv(index=False).encode("utf-8"),
                               "sentiment_predictions.csv", "text/csv")
