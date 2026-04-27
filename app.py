import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import json
import joblib
import google.generativeai as genai  # Gemini API

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
FEATURE_EXTRACTOR_PATH = 'feature_extractor.h5'
CLASSIFIER_PATH = 'random_forest_classifier.joblib'
CLASS_INDICES_PATH = 'class_indices.json'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Gemini API ---
GEMINI_API_KEY = "AIzaSyDATrnMbi2SpQrJBTRDN77v-sQX1HkA-ks"
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

# --- Load Models ---
try:
    print(f"Loading Keras feature extractor from {FEATURE_EXTRACTOR_PATH}...")
    feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
    print("Feature extractor loaded successfully.")

    print(f"Loading Random Forest classifier from {CLASSIFIER_PATH}...")
    classifier = joblib.load(CLASSIFIER_PATH)
    print("Classifier loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    feature_extractor = None
    classifier = None

# --- Load Class Indices ---
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}
    print("Class indices loaded successfully.")
except Exception as e:
    print(f"Error loading class indices: {e}")
    class_labels = None


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_disease(img_path):
    """
    Extracts features from an image and uses the classifier to predict the disease.
    """
    if feature_extractor is None or classifier is None or class_labels is None:
        return "Models or class labels not loaded", 0.0

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_array)

        # Feature extraction
        features = feature_extractor.predict(processed_img, verbose=0)

        # Random Forest prediction
        prediction_index = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        confidence = np.max(probabilities)
        predicted_label = class_labels[prediction_index]

        return predicted_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction", 0.0


def get_gemini_response(predicted_label, confidence):
    """
    Sends the predicted result to Gemini to get an industry use-case suggestion,
    and formats the output for clean multiline HTML display.
    """
    try:
        prompt = (
            f"The AI disease detection model predicted '{predicted_label}' "
            f"with {confidence * 100:.2f}% confidence.\n"
            f"Suggest the top 3 real-world industries or business sectors "
            f"that could benefit from this leaf detection system.\n"
            f"Provide me just 3 industry names and nothing else. "
            f"Make it in 3 bullet points, one per line. "
            f"Don't bold any text or use asterisks (*)."
        )

        response = model_gemini.generate_content(prompt)
        text = response.text.strip()

        # ✅ Ensure each bullet is shown on its own line in HTML
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        formatted_text = "<br>".join(lines)

        return formatted_text

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Could not fetch industry insights."


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predicted_label, confidence = predict_disease(filepath)

            # --- Get Gemini insights ---
            industry_insights = get_gemini_response(predicted_label, confidence)

            return render_template(
                'index.html',
                prediction=f'{predicted_label}',
                confidence=f'{confidence*100:.2f}%',
                uploaded_image=filename,
                gemini_response=industry_insights
            )

    return render_template('index.html', prediction=None, confidence=None, uploaded_image=None, gemini_response=None)


@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Main ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
