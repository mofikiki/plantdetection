import os
import subprocess
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import cv2
import time
from datetime import datetime, timedelta

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the labels
with open("converted_keras/labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Path to the folder where you want to save images
output_folder = "/Users/mofi/Downloads/abnormal_images/"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the model
model = load_model("converted_keras/keras_model.h5", compile=False)

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Function to preprocess and predict an image
def predict_image(image):
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = cv2.resize(image, size)
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Turn the image into a numpy array
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)  # Normalize pixels to [0, 1]
    # Load the image into the array
    data[0] = normalized_image_array
    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]  # Class name without stripping whitespace
    confidence_score = prediction[0][index]
    if class_name == "healthy" and confidence_score < 0.8:
        class_name = "abnormality"
    return class_name, confidence_score

# Function to activate hotspot
def activate_hotspot():
    hotspot_name = "MyHotspot"
    ssid_name = "MySSID"
    password = "MyPassword"
    subprocess.run(["sudo", "nmcli", "device", "wifi", "hotspot", "con-name", hotspot_name, "ssid", ssid_name, "password", password])

# Function to get system IP address
def get_system_ip():
    ip_result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
    ip_address = ip_result.stdout.strip()
    return ip_address

# Activate the hotspot
activate_hotspot()

# Get system IP address
ip_address = get_system_ip()
print(f"Hotspot activated. System IP address: {ip_address}")

# Access the camera
camera = cv2.VideoCapture(0)

# Set initial time for capturing
start_time = time.time()

# Flag to indicate whether an abnormal image has been sent
abnormal_sent = False

while not abnormal_sent:
    # Capture frame-by-frame every 5 seconds
    if time.time() - start_time >= 5:
        # Reset the start time for next capture
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Perform prediction on the captured frame
        predicted_class, confidence = predict_image(frame)

        # Skip saving if the predicted class is "healthy"
        if predicted_class == "healthy":
            continue

        # Create the class-specific output folder if it doesn't exist
        class_output_folder = os.path.join(output_folder, predicted_class)
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)

        # Save the image to the class-specific output folder
        image_filename = os.path.join(class_output_folder, f"{predicted_class}_{confidence:.4f}.jpg")
        cv2.imwrite(image_filename, frame)

        print(f"Image saved as {image_filename}")

        # Check if the predicted class is "abnormality" and set abnormal_sent flag to True
        if predicted_class == "abnormality":
            abnormal_sent = True

    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Delete files older than 7 days
now = datetime.now()
for root, dirs, files in os.walk(output_folder):
    for file in files:
        file_path = os.path.join(root, file)
        creation_time = datetime.fromtimestamp(os.stat(file_path).st_ctime)
        if now - creation_time > timedelta(days=7):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
