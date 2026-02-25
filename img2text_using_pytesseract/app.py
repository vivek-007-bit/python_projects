from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import os
import uuid
import shutil
import time

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

def preprocess_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((1200, 1200))
    img = img.convert("L")
    return img

@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = None
    processing_time = None

    if request.method == "POST":
        start_time = time.perf_counter()

        file = request.files.get("image")

        if file and file.filename != "":
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            try:
                img = preprocess_image(filepath)

                extracted_text = pytesseract.image_to_string(
                    img,
                    config="--psm 6",
                    timeout=20
                )

            except RuntimeError:
                extracted_text = "Processing took too long. Try a smaller image."

            except Exception as e:
                extracted_text = f"Error processing image: {str(e)}"

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        end_time = time.perf_counter()
        processing_time = round(end_time - start_time, 2)

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

