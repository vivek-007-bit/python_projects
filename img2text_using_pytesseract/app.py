from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import uuid
import shutil

app = Flask(__name__)

# TESSERACT PATH 
tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# UPLOAD CONFIG
UPLOAD_FOLDER = "static/uploads"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# IMAGE PREPROCESSING 
def preprocess_image(path):
    """
    Reduce dimensionality + improve handwriting OCR.
    Prevents Render timeout & memory crash.
    """

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Invalid image file")

    #Resize 
    h, w = img.shape[:2]
    max_dim = 1200

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Noise removal
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Convert back to PIL image 
    pil_img = Image.fromarray(thresh)

    return pil_img


# ROUTES
@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:

            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            try:
                # preprocess
                processed_img = preprocess_image(filepath)

                # OCR (multi-language)
                extracted_text = pytesseract.image_to_string(
                    processed_img,
                    lang="eng+hin+jpn",
                    config="--oem 1 --psm 6",
                    timeout=20
                )

            except RuntimeError:
                extracted_text = "OCR timeout. Try smaller or clearer image."

            except Exception as e:
                extracted_text = f"Error: {str(e)}"

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    return render_template(
        "index.html",
        extracted_text=extracted_text
    )


# RUN SERVER
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
