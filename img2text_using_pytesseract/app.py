from flask import Flask, render_template, request
import pytesseract
import cv2
import numpy as np
import os
import uuid
import shutil
import time

os.environ["OMP_THREAD_LIMIT"] = "1"

cv2.setUseOptimized(True)
cv2.setNumThreads(1)

app = Flask(__name__)

tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

UPLOAD_FOLDER = "static/uploads"
MAX_FILE_SIZE = 5 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def normalize_size(img):
    h, w = img.shape[:2]
    max_side = 1100

    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    return img


def crop_text_region(binary):
    inv = 255 - binary
    coords = cv2.findNonZero(inv)

    if coords is None:
        return binary

    x, y, w, h = cv2.boundingRect(coords)

    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)

    return binary[y:y + h + pad, x:x + w + pad]


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded")

    img = normalize_size(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bg = cv2.GaussianBlur(gray, (21, 21), 0)
    gray = cv2.addWeighted(gray, 1.5, bg, -0.5, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    processed = cv2.GaussianBlur(enhanced, (3, 3), 0)

    variance = processed.std()

    if variance < 45:
        mode = "handwritten"

        thresh = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            8
        )
    else:
        mode = "printed"

        _, thresh = cv2.threshold(
            processed,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    thresh = crop_text_region(thresh)

    thresh = cv2.copyMakeBorder(
        thresh,
        10,
        10,
        10,
        10,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return thresh, mode


def build_tesseract_config(mode):
    psm = 4 if mode == "handwritten" else 6

    return (
        f"--oem 1 --psm {psm} "
        "-c load_system_dawg=0 "
        "-c load_freq_dawg=0 "
        "-c preserve_interword_spaces=1"
    )


@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = None
    processing_time = None

    if request.method == "POST":
        start_time = time.perf_counter()

        file = request.files.get("image")

        if file and file.filename:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            try:
                processed_img, mode = preprocess_image(filepath)

                config = build_tesseract_config(mode)

                extracted_text = pytesseract.image_to_string(
                    processed_img,
                    lang="eng",
                    config=config,
                    timeout=15
                )

            except RuntimeError:
                extracted_text = "Processing timeout. Try smaller image."

            except Exception as e:
                extracted_text = f"Error: {str(e)}"

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        processing_time = round(time.perf_counter() - start_time, 2)
        print(f"Task completed in {processing_time} seconds")

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        processing_time=processing_time
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )