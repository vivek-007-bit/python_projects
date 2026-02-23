from flask import Flask, render_template, request
import joblib
import os
from preprocess import preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load("handwriting_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = ""

    if request.method == "POST":

        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        img = preprocess_image(path)

        pred = model.predict(img)[0]

        prediction = f"Predicted Character: {pred}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)