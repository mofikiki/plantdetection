import os
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the labels
with open("converted_keras/labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Define the path to the image or folder containing images
path = "/Users/mofi/Desktop/dfb4b21f-48d7-4177-a855-5906ab42eb09.jpeg"  # Change this to your path

# Determine if the path is a directory or a single file
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
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
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

# Process images in the specified path
if os.path.isdir(path):
    # Iterate through images in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpeg", ".jpg", ".png")):
            image_path = os.path.join(image_folder, filename)
            predicted_class, confidence = predict_image(image_path)
            # Create the class-specific output folder if it doesn't exist
            class_output_folder = os.path.join(output_folder, predicted_class)
            if not os.path.exists(class_output_folder):
                os.makedirs(class_output_folder)
            # Move the image to the class-specific output folder
            output_path = os.path.join(class_output_folder, filename)
            try:
                os.rename(image_path, output_path)
                print(f"Image {filename} is predicted as {predicted_class} with confidence {confidence}. Moved to {class_output_folder}.")
            except Exception as e:
                print(f"Failed to move image {filename}: {e}")
else:
    # Process a single image
    predicted_class, confidence = predict_image(path)
    # Skip healthy class
    if predicted_class == "2 Healthy":
        print(f"Image is predicted as healthy with confidence {confidence}. Not moved.")
    else:
        # Create the class-specific output folder if it doesn't exist
        class_output_folder = os.path.join(output_folder, predicted_class)
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)
        # Move the image to the class-specific output folder
        output_path = os.path.join(class_output_folder, os.path.basename(path))
        try:
            os.rename(path, output_path)
            print(f"Image is predicted as {predicted_class} with confidence {confidence}. Moved to {class_output_folder}.")
        except Exception as e:
            print(f"Failed to move image: {e}")
