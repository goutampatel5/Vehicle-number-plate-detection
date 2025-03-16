import cv2
import pytesseract
import torch
from pathlib import Path
import sys
import os

# Add the path to YOLOv7 repository if it's not installed as a package
sys.path.append('D:/ml_project/yolov7')  # Update this path to the YOLOv7 directory
from models.experimental import attempt_load  # YOLOv7-specific load function
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Create a directory to save cropped images (if it doesn't exist)
output_dir = 'cropped_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load YOLOv7 model using PyTorch
def load_yolo_model(model_path):
    print("Loading YOLO model...")
    device = select_device('cuda')  # or 'cuda' if GPU is available
    model = attempt_load(model_path, map_location=device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")
    return model, device

# Function to apply YOLO, crop number plates, and then apply OCR
def detect_and_ocr(image_path, model, device):
    print("Script started...")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or unable to read.")
        return
    
    # Display original image to confirm it's being read correctly
    print(f"Image shape: {image.shape}")
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert image to RGB as YOLO expects RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare image for YOLO model
    img = torch.from_numpy(rgb_image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img = img.to(device)
    
    # Run YOLO detection on the image
    pred = model(img)[0]
    print(f"Prediction output: {pred}")
    
    detections = non_max_suppression(pred, 0.25, 0.45)
    print(f"Number of detections: {len(detections)}")
    
    if not detections:
        print("No detections found.")
        return
    
    extracted_texts = []  # List to store OCR results
    
    # Process each detection
    for det in detections:  # detections per image
        if len(det):
            for *box, conf, class_id in det:  # xyxy format bounding box
                print(f"Bounding box: {box}, Confidence: {conf}")
                x1, y1, x2, y2 = map(int, box)
                
                # Crop the detected number plate from the image
                number_plate = image[y1:y2, x1:x2]
                
                # Save the cropped number plate image
                cropped_image_path = os.path.join(output_dir, f"cropped_plate_{x1}_{y1}.jpg")
                cv2.imwrite(cropped_image_path, number_plate)  # Save image
                print(f"Saved cropped number plate to: {cropped_image_path}")
                
                # Convert the cropped image to grayscale for better OCR results
                gray_plate = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)
                
                # Optional: Apply thresholding to improve OCR accuracy
                gray_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Use pytesseract to extract text from the cropped image
                extracted_text = pytesseract.image_to_string(gray_plate, config='--psm 8')
                print(f"Extracted text: {extracted_text}")
                extracted_texts.append(extracted_text.strip())  # Add text to the list
                
                # Display cropped number plate for verification (optional)
                cv2.imshow("Cropped Number Plate", number_plate)
                cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # Print all extracted texts
    for i, text in enumerate(extracted_texts):
        print(f"Extracted Text from Number Plate {i+1}: {text}")
    
    print("Script finished...")

# Load the YOLO model and run detection
model_path = "D:/ml_project/yolov7/runs/train/exp/weights/best.pt"  # Correct model path
yolo_model, device = load_yolo_model(model_path)
image_path = "C:/Users/patel/Downloads/car.jpg"
detect_and_ocr(image_path, yolo_model, device)
