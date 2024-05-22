import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the labels
class_names = open("converted_keras/labels.txt", "r").readlines()

# Load the model
model = load_model("converted_keras/keras_model.h5", compile=False)

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Function to preprocess and predict an image
def predict_image(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL format
    pil_image = Image.fromarray(image_rgb)
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image_resized = ImageOps.fit(pil_image, size, Image.ANTIALIAS)
    # Turn the image into a numpy array
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)  # Normalize pixels to [0, 1]
    # Load the image into the array
    data[0] = normalized_image_array
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove leading/trailing whitespaces
    confidence_score = prediction[0][index]
    if class_names[index].strip()=="healthy" and confidence<0.8:
        class_name=="abnormality"
    else:
        class_name= class_names[index].strip()
    return class_name, confidence_score

# Path to the folder where you want to save images
output_folder = "path/to/your/output_folder/"
# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Access camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Predict the image
    predicted_class, confidence = predict_image(frame)

    # Check if the predicted class is not "healthy"
    if predicted_class == ['abnormality','rust','powdery']:
        print(f"Object is predicted as {predicted_class} with confidence {confidence}.")
        # Save the image to the output folder
        image_name = f"unhealthy_{int(time.time())}.jpg"  # Unique filename based on current time
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, frame)
        print(f"Image saved as {output_path}.")
    else:
        print(f"Object is predicted as {predicted_class} with confidence {confidence}.")

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyALLWindows()
