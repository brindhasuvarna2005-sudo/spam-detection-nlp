from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import re
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# -----------------------------
# LOAD MODELS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model/hybrid_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "model/scaler.pkl"), "rb"))

embedder = SentenceTransformer('all-MiniLM-L6-v2')


# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"\d+", "NUM", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(text):
    return np.array([
        len(text),
        sum(c.isdigit() for c in text),
        sum(c.isupper() for c in text),
        len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', text)),
        len(re.findall(r'http\S+', text)),
        len(text.split()),
        0,  # placeholder for source encoding
        int(bool(re.search(r"http|www", text))),
        int(bool(re.search(r"₹|\$|£|€", text))),
        int(bool(re.search(r"free|win|winner|urgent|offer|claim|prize|cash", text, re.I))),
        sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    ])


# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        # 🔥 MULTI-INPUT SUPPORT
        input_type = data.get("type", "sms")  # sms or email
        message = data["message"]
        subject = data.get("subject", "")

        # Combine subject + body for email
        if input_type == "email":
            full_text = subject + " " + message
        else:
            full_text = message

        # -----------------------------
        # PREPROCESS
        # -----------------------------
        cleaned = clean_text(full_text)

        # Embedding
        text_vec = embedder.encode([cleaned])

        # Metadata
        meta = extract_features(full_text).reshape(1, -1)
        meta_scaled = scaler.transform(meta)

        # Combine features
        hybrid_input = np.hstack((text_vec, meta_scaled))

        # -----------------------------
        # PREDICTION
        # -----------------------------
        prediction = int(model.predict(hybrid_input)[0])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(hybrid_input)[0]
            ham_prob = float(probs[0])
            spam_prob = float(probs[1])
        else:
            ham_prob = float(1 - prediction)
            spam_prob = float(prediction)

        label = "Spam" if prediction == 1 else "Ham"

        # -----------------------------
        # RESPONSE
        # -----------------------------
        return jsonify({
            "label": label,
            "spam_confidence": round(spam_prob * 100, 2),
            "ham_confidence": round(ham_prob * 100, 2),
            "type": input_type
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)