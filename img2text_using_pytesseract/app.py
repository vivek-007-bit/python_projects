from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import os
import uuid
import shutil

app = Flask(__name__)

# TESSERACT PATH CONFIGURATION
tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Windows fallback (local development)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# UPLOAD CONFIGURATION
UPLOAD_FOLDER = "static/uploads"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# IMAGE PREPROCESSING 
def preprocess_image(image_path):
    """
    Resize and simplify image to reduce OCR load.
    Prevents Render timeout & memory errors.
    """
    img = Image.open(image_path)

    # Resize large images
    img.thumbnail((1200, 1200))

    # Convert to grayscale
    img = img.convert("L")

    return img

# ROUTES
@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":

            # Create unique filename 
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            try:
                # Preprocess image
                img = preprocess_image(filepath)

                # OCR with timeout
                extracted_text = pytesseract.image_to_string(
                    img,
                    lang="eng+jpn+hin",
                    config="--psm 6",
                    timeout=20
                )

            except RuntimeError:
                extracted_text = "Processing took too long. Try a smaller image."

            except Exception as e:
                extracted_text = f"Error processing image: {str(e)}"

            finally:
                # Clean up uploaded file 
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

