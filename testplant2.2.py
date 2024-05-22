import os
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the labels
class_names = open("converted_keras/labels.txt", "r").readlines()

# Define the path to the image or folder containing images
path = "/Users/mofi/Downloads/sugarcane/Healthy/healthy (36).jpeg"  # Change this to your path

# If the path is a directory, process all images in the directory
if os.path.isdir(path):
    image_folder = path
else:
    # If the path is a file, process the single image
    image_folder = os.path.dirname(path)

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
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # Turn the image into a numpy array
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)  # Normalize pixels to [0, 1]
    # Load the image into the array
    data[0] = normalized_image_array
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove leading/trailing whitespaces
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# If the path is a directory, iterate through images in the folder
if os.path.isdir(path):
    # Iterate through images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            predicted_class, confidence = predict_image(image_path)
            # Check if the predicted class is not "healthy"
            if predicted_class != "healthy":
                # Move the image to the output folder
                output_path = os.path.join(output_folder, filename)
                os.rename(image_path, output_path)
                print(f"Image {filename} is predicted as {predicted_class} with confidence {confidence}. Moved to {output_folder}.")
            else:
                print(f"Image {filename} is predicted as {predicted_class} with confidence {confidence}.")
else:
    # If the path is a file, process the single image
    image_path = path
    predicted_class, confidence = predict_image(image_path)
    # Check if the predicted class is not "healthy"
    if predicted_class != "healthy":
        # Move the image to the output folder
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        os.rename(image_path, output_path)
        print(f"Image {os.path.basename(image_path)} is predicted as {predicted_class} with confidence {confidence}. Moved to {output_folder}.")
    else:
        print(f"Image {os.path.basename(image_path)} is predicted as {predicted_class} with confidence {confidence}.")
