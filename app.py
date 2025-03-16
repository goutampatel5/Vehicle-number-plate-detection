from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import pytesseract
import torch
from pathlib import Path
import sys
sys.path.append('D:/ml_project/yolov7')
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from werkzeug.utils import secure_filename
import os

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve images from the upload folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Load YOLO model function
def load_yolo_model(model_path):
    device = select_device('')  # Select GPU if available
    model = attempt_load(model_path, map_location=device)  # Load YOLOv7 model
    model.eval()
    return model, device

# Load YOLO model
model_path = "D:/ml_project/yolov7/runs/train/exp/weights/best.pt"  # Update the path
yolo_model, device = load_yolo_model(model_path)

# Updated detect_and_ocr function
def detect_and_ocr(image_path, model, device):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None, None, None

    # Get original dimensions of the image
    orig_h, orig_w, _ = image.shape

    # Convert image to RGB for YOLO compatibility
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image for YOLO input size
    input_size = 640
    img_resized = cv2.resize(rgb_image, (input_size, input_size))

    # Prepare image tensor for YOLO model
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # Run YOLO inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Apply Non-Maximum Suppression
    detections = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)

    # Scaling factors to map bounding boxes back to the original image size
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    extracted_texts = []
    cropped_images = []

    for det in detections:
        if det is not None and len(det):
            for *box, conf, cls in det:  # Each detection includes bbox, confidence, and class
                # Scale bounding box coordinates back to original image size
                x1, y1, x2, y2 = map(int, box)
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # Add padding to the bounding box
                padding_w = int((x2 - x1) * 0.01)  # 5% width padding
                padding_h = int((y2 - y1) * 0.01)  # 5% height padding
                x1 = max(0, x1 - padding_w)
                y1 = max(0, y1 - padding_h)
                x2 = min(orig_w, x2 + padding_w)
                y2 = min(orig_h, y2 + padding_h)

                # Crop the detected number plate
                cropped_plate = image[y1:y2, x1:x2]
                if cropped_plate.size == 0:
                    continue
                cropped_images.append(cropped_plate)

                # Debugging: Draw the bounding box on the original image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Preprocess cropped image for OCR
                gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

                # Perform OCR
                ocr_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                extracted_text = pytesseract.image_to_string(gray_plate, config=ocr_config)
                clean_text = ''.join(filter(str.isalnum, extracted_text)).upper()
                extracted_texts.append(clean_text)
                print(f"Extracted text: {clean_text}")  # Print extracted text for debugging

    return image, extracted_texts, cropped_images

# Flask route to handle the home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run detection and OCR
    detected_image, extracted_texts, cropped_images = detect_and_ocr(filepath, yolo_model, device)
    if not extracted_texts:
        return jsonify({'error': 'No number plates detected'}), 400

    # Save detected image with bounding boxes
    detected_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'detected_{filename}')
    cv2.imwrite(detected_image_path, detected_image)

    # Save cropped images
    cropped_paths = []
    for i, cropped in enumerate(cropped_images):
        cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], f'crop_{i}_{filename}')
        cv2.imwrite(cropped_path, cropped)
        cropped_paths.append(cropped_path)

    return jsonify({
        'number_plates': extracted_texts,
        'detected_image': detected_image_path,
        'cropped_images': cropped_paths
    })

if __name__ == '__main__':
    app.run(debug=True)
