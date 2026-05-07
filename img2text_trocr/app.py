import gradio as gr
from PIL import Image
import torch
import cv2
import numpy as np
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

#loading the model
MODEL_ID = 'trocr_model'

processor = TrOCRProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if device == "cuda":
    model = model.half()


#normalizing the height
def normalize_height(img, target_h=64):
    h, w = img.shape[:2]
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))


#text cleaning to remove the noise
def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        #removing common noise
        line = re.sub(r'\b0\s+0\b', '', line)
        line = re.sub(r'\b\d{3,4}\b', '', line)
        line = re.sub(r'\b[a-zA-Z]\b', '', line)

        #removing weird characters but keep punctuation
        line = re.sub(r'[^\w\s,.!?-]', '', line)

        #normalizing spacing within the lines
        line = re.sub(r'\s+', ' ', line)

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def is_valid_line(line):
    if len(line.strip()) < 3:
        return False

    digit_ratio = sum(c.isdigit() for c in line) / len(line)
    if digit_ratio > 0.5:
        return False

    special_ratio = sum(not c.isalnum() and not c.isspace() for c in line) / len(line)
    if special_ratio > 0.4:
        return False

    return True


def fix_punctuation(text):
    lines = text.split("\n")
    fixed = []

    for line in lines:
        line = re.sub(r'\s+,', ',', line)
        line = re.sub(r'\s+\.', '.', line)
        fixed.append(line)

    return "\n".join(fixed)


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

    
    #image preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    #deskewing the image
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

    progress(0.4, desc="Detecting text lines...")

    
    #text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w < 40 or h < 15:
            continue
        if w * h < 1200:
            continue

        raw_boxes.append((x, y, w, h))

    raw_boxes = sorted(raw_boxes, key=lambda b: b[1])

    

    #grouping the extracted text in a common line
    lines = []
    current_line = []
    line_threshold = 15

    for box in raw_boxes:
        x, y, w, h = box

        if not current_line:
            current_line.append(box)
            continue

        _, prev_y, _, _ = current_line[-1]

        if abs(y - prev_y) < line_threshold:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    if current_line:
        lines.append(current_line)

    #merging the lines
    boxes = []

    for line in lines:
        xs = [b[0] for b in line]
        ys = [b[1] for b in line]
        ws = [b[0] + b[2] for b in line]
        hs = [b[1] + b[3] for b in line]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(ws)
        y_max = max(hs)

        boxes.append((y_min, x_min, x_max - x_min, y_max - y_min))

    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))

    #drawing bounding boxes
    vis_image = image.copy()
    for (y, x, w, h) in boxes:
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    progress(0.7, desc="Running OCR...")

    #calling the ocr app
    results = []
    crops = []

    for (y, x, w, h) in boxes:
        pad = 10

        crop = image[
            max(0, y - pad):y + h + pad,
            max(0, x - pad):x + w + pad
        ]

        if np.sum(crop > 0) < 500:
            continue

        if crop.shape[1] < crop.shape[0] * 1.2:
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

    #preserving the actual text layout
    filtered = [line for line in results if is_valid_line(line)]
    final_text = "\n".join(filtered)
    final_text = clean_text(final_text)
    final_text = fix_punctuation(final_text)

    progress(1.0, desc="Done")

    output_image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))

    return output_image, final_text


#Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("TrOCR | Handwriting to text")

    with gr.Column():
        input_image = gr.Image(type="pil", label="Upload Image")
        submit_btn = gr.Button("Run OCR")

    with gr.Row():
        output_image = gr.Image(
            label="Detected Text Lines",
            scale=1
        )

        output_text = gr.Textbox(
            label="Extracted Text",
            lines=15,
            scale=1
        )

    submit_btn.click(
        fn=ocr_pipeline,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

demo.launch()
