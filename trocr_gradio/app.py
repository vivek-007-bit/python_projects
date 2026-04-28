import gradio as gr
from PIL import Image
import torch
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

#loading the model
MODEL_ID = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if device == "cuda":
    model = model.half()


#normalize height
def normalize_height(img, target_h=64):
    h, w = img.shape[:2]
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))


#ocr pipeline
def ocr_pipeline(pil_image, progress=gr.Progress()):
    if pil_image is None:
        return None, "Upload an image"

    progress(0.1, desc="Preprocessing image...")

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    MAX_WIDTH = 1200
    h, w = image.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        image = cv2.resize(image, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Deskew
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        gray = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

        image = cv2.warpAffine(image, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh1 = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    _, thresh2 = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    thresh = cv2.bitwise_or(thresh1, thresh2)

    kernel_clean = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)

    progress(0.4, desc="Detecting text regions...")

    #text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 15:
            continue
        if w * h < 1200:
            continue
        boxes.append((y, x, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    #draw bounding boxes
    vis_image = image.copy()
    for (y, x, w, h) in boxes:
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    progress(0.7, desc="Running OCR...")

    #calling the OCR 
    results = []

    if not boxes:
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(pixel_values)

        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return pil_image, text

    crops = []

    for (y, x, w, h) in boxes:
        pad = 10
        crop = image[
            max(0, y - pad):y + h + pad,
            max(0, x - pad):x + w + pad
        ]

        if crop.shape[1] < crop.shape[0]:
            continue

        crop = normalize_height(crop)
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crops.append(pil_crop)

    if crops:
        pixel_values = processor(
            images=crops,
            return_tensors="pt",
            padding=True
        ).pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=64)

        texts = processor.batch_decode(ids, skip_special_tokens=True)
        results = [t.strip() for t in texts if t.strip()]

    final_text = "\n".join(results)

    progress(1.0, desc="Done")

    # Convert OpenCV image back to PIL for display
    output_image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))

    return output_image, final_text


#GRADIO UI
with gr.Blocks() as demo:
    gr.Markdown("Handwriting OCR (TrOCR)")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image")
        output_image = gr.Image(label="Detected Regions")

    output_text = gr.Textbox(label="Extracted Text", lines=15)

    submit_btn = gr.Button("Run OCR")

    submit_btn.click(
        fn=ocr_pipeline,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

demo.launch()