import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

model = YOLO(os.path.join(BASE_DIR, "best.pt"))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected. Please choose a file."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"error": "Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP."}
        ), 400

    ext = Path(file.filename).suffix.lower()
    unique_stem = uuid.uuid4().hex
    unique_name = unique_stem + ext
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        file.save(upload_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded file: {e}"}), 500

    try:
        results = model.predict(
            source=upload_path,
            save=True,
            project=os.path.join(BASE_DIR, "static"),
            name="results",
            exist_ok=True,
            conf=0.25,
        )

        # ultralytics saves output with the same basename as the input
        result_path = os.path.join(RESULTS_FOLDER, unique_name)

        # Fallback: YOLO sometimes converts PNG → JPG on save
        if not os.path.exists(result_path):
            for alt_ext in (".jpg", ".png", ".jpeg"):
                candidate = os.path.join(RESULTS_FOLDER, unique_stem + alt_ext)
                if os.path.exists(candidate):
                    result_path = candidate
                    break

        if not os.path.exists(result_path):
            return jsonify({"error": "Prediction output image not found."}), 500

        num_detections = (
            len(results[0].boxes) if results[0].boxes is not None else 0
        )
        labels = []
        if results[0].boxes is not None and results[0].names:
            class_ids = results[0].boxes.cls.tolist()
            labels = list({results[0].names[int(c)] for c in class_ids})

        return jsonify(
            {
                "original": f"/static/uploads/{Path(upload_path).name}",
                "predicted": f"/static/results/{Path(result_path).name}",
                "detections": num_detections,
                "labels": labels,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
