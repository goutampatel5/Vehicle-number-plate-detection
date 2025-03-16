import os
import subprocess

# Paths to your dataset and YOLOv7 repository
dataset_path = '/home/shivamp22co/ml_project/dataset'  # Corrected path
yolov7_repo_path = '/home/shivamp22co/ml_project/yolov7'
train_script_path = os.path.join(yolov7_repo_path, 'train.py')
test_script_path = os.path.join(yolov7_repo_path, 'test.py')
detect_script_path = os.path.join(yolov7_repo_path, 'detect.py')

# Set your parameters for training
batch_size = 16
epochs = 300
img_size = 640  # Image size for training and detection
data_file = os.path.join("/home/shivamp22co/ml_project/", 'dataset.yaml')  # YOLO dataset config file
weights = 'yolov7.pt'  # Pre-trained weights file (for fine-tuning) or ''''' for training from scratch

# 1. Training YOLOv7
def train_yolov7():
    train_command = [
        'python3', train_script_path,
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', data_file,
        '--weights', weights,  # Leave blank to train from scratch
        '--device', '0'  # Specify GPU id, change to 'cpu' if no GPU
    ]

    try:
        # Navigate to YOLOv7 repo
        os.chdir(yolov7_repo_path)

        # Start the training process
        subprocess.run(train_command, check=True)
        print("Training started successfully!")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

# 2. Testing YOLOv7
def test_yolov7():
    test_command = [
        'python3', test_script_path,
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--data', data_file,
        '--weights', os.path.join(yolov7_repo_path, 'runs', 'train', 'exp', 'weights', 'best.pt'),  # Use best weights from training
        '--device', '0'  # Change to 'cpu' if no GPU
    ]

    try:
        # Navigate to YOLOv7 repo
        os.chdir(yolov7_repo_path)

        # Start the testing process
        subprocess.run(test_command, check=True)
        print("Testing completed successfully!")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during testing: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

# 3. Detect using YOLOv7
def detect_yolov7(source_folder):
    detect_command = [
        'python3', detect_script_path,
        '--weights', os.path.join(yolov7_repo_path, 'runs', 'train', 'exp', 'weights', 'best.pt'),  # Use the best weights from training
        '--img', str(img_size),
        '--conf', '0.25',  # Confidence threshold for detection
        '--source', source_folder,  # Source folder for images/videos
        '--device', '0'  # Change to 'cpu' if no GPU
    ]

    try:
        # Navigate to YOLOv7 repo
        os.chdir(yolov7_repo_path)

        # Start the detection process
        subprocess.run(detect_command, check=True)
        print("Detection completed successfully!")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during detection: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

if __name__ == '__main__':
    # First, train the model
    print("Starting training...")
    train_yolov7()

    # After training, test the model
    print("\nStarting testing...")
    test_yolov7()

    # Finally, detect objects in a new set of images (e.g., test images)
    print("\nStarting detection...")
    source_folder = '/home/shivamp22co/ml_project/dataset/images/test'  # Corrected folder path
    detect_yolov7(source_folder)
