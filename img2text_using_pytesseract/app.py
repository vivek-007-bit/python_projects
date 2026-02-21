from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import os

app = Flask(__name__)

#SET TESSERACT PATH 
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_PATH", "/usr/bin/tesseract"
)

#Upload configuration
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():

    extracted_text = None

    print("Request Method:", request.method)

    # When form submitted
    if request.method == "POST":

        # Get uploaded image
        file = request.files.get("image")
        print("File received:", file)

        if file and file.filename != "":

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            # Save image
            file.save(filepath)
            print("Saved at:", filepath)

            try:
                # Open image
                img = Image.open(filepath)

                # OCR Extraction
                extracted_text = pytesseract.image_to_string(img)

                print("Extracted Text Length:",
                      len(extracted_text) if extracted_text else 0)

            except Exception as e:
                extracted_text = f"Error processing image: {str(e)}"
                print(extracted_text)

    # Send result to template
    return render_template(
        "index.html",
        extracted_text=extracted_text
    )
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))


