from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import cv2
import os
import uuid
import shutil
import threading

app = Flask(__name__)

tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

results = {}

def preprocess_image(path):
    img = cv2.imread(path)

    h, w = img.shape[:2]
    max_dim = 700

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(thresh)

def run_ocr(task_id, filepath):
    try:
        img = preprocess_image(filepath)

        text = pytesseract.image_to_string(
            img,
            lang="eng",
            config="--oem 3 --psm 7"
        )

        results[task_id] = text

    except Exception as e:
        results[task_id] = f"Error: {str(e)}"

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            task_id = uuid.uuid4().hex
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                f"{task_id}_{file.filename}"
            )

            file.save(filepath)

            results[task_id] = "Processing..."

            threading.Thread(
                target=run_ocr,
                args=(task_id, filepath)
            ).start()

            return render_template(
                "index.html",
                extracted_text="Processing... refresh page in a few seconds."
            )

    return render_template("index.html", extracted_text=None)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
