
import io
import json
from pathlib import Path
from html import escape

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

# Project files
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "potato_disease_model.keras"
CLASS_PATH = BASE_DIR / "class_names.json"

# Check required files
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CLASS_PATH.is_file():
    raise FileNotFoundError(f"Class names not found: {CLASS_PATH}")

# Load model and class names once when the app starts
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

if len(CLASS_NAMES) != 3:
    raise ValueError("Expected exactly three class names.")

app = FastAPI(title="Potato Leaf Disease Classifier")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Potato Leaf Disease Classifier</title>
    <style>
        body {
            margin: 0;
            padding: 40px 16px;
            font-family: Arial, sans-serif;
            background: #f0f7f1;
            color: #243328;
        }
        .card {
            max-width: 600px;
            margin: auto;
            padding: 30px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 5px 22px #00000012;
        }
        h1 { color: #237443; }
        input { margin: 20px 0; max-width: 100%; }
        button {
            background: #237443;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 22px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background: #185a32; }
        .result {
            margin-top: 24px;
            padding: 18px;
            background: #edf7ee;
            border-radius: 10px;
        }
        .error { background: #fff0ed; }
        .muted { color: #65736a; font-size: 14px; }
    </style>
</head>
<body>
    <main class="card">
        <h1>Potato Leaf Disease Detection</h1>
        <p>Upload a potato leaf image to classify it as early blight,
        late blight, or healthy.</p>

        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/jpeg,image/png,image/webp"
                   required>
            <br>
            <button type="submit">Predict Disease</button>
        </form>

        <div id="result">{result}</div>
        <p class="muted">Predictions are model estimates, not a definitive
        agricultural diagnosis.</p>
    </main>
</body>
</html>
"""


def render_page(result=""):
    # Replace only our specific placeholder, not the CSS braces.
    return HTML_PAGE.replace("{result}", result)


@app.get("/", response_class=HTMLResponse)
async def home():
    return render_page()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "classes": CLASS_NAMES
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG, PNG, or WebP image."
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    try:
        with Image.open(io.BytesIO(contents)) as img:
            image = img.convert("RGB").resize((224, 224))

    except (UnidentifiedImageError, OSError):
        return render_page(
            '<div class="result error">Unable to read the image. '
            'Please try another file.</div>'
        )

    # Match training: input [0, 1].
    # The saved model includes Rescaling to [-1, 1].
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    readable_name = predicted_class.replace("Potato___", "").replace("_", " ")

    result_html = f"""
    <section class="result">
        <h2>Prediction: {escape(readable_name)}</h2>
        <p>Model confidence: {confidence:.2f}%</p>
        <p class="muted">Filename: {escape(file.filename or "uploaded image")}</p>
    </section>
    """

    return render_page(result_html)
