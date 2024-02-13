import os
import numpy as np
from PIL import Image
from keras.models import load_model

# Define the class labels based on your dataset
class_labels = ["Mild", "Severe", "Moderate", "Proliferate_DR", "No_DR"]

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(180, 200), convert_to_grayscale=True, normalize_pixels=True):
    image = Image.open(image_path)
    image = image.resize(target_size)

    if convert_to_grayscale:
        image = image.convert('L')

    if normalize_pixels:
        image = np.array(image) / 255.0

    return image

# Load the trained model
model = load_model(r'C:\Users\PC-ACER\PycharmProjects\pythonProject\ModelTraining.h5')

# Example usage for a single image
input_image_path = r'C:\Users\PC-ACER\Downloads\gaussian_filtered_images\gaussian_filtered_images\Mild\f7fec8935126.png'  # Specify the actual image file
input_image = preprocess_image(input_image_path)
input_image = input_image.reshape(1, 180, 200, 1)  # Reshape for model input

# Make a prediction
predicted_probabilities = model.predict(input_image)
predicted_class = np.argmax(predicted_probabilities, axis=1)
predicted_class_name = class_labels[predicted_class[0]]

# Display the predicted class and probabilities
print(f'Predicted Class: {predicted_class_name}')
print(f'Predicted Probabilities: {predicted_probabilities[0]}')

# For demonstration purposes, let's assume the ground truth class is 4 (No_DR)
ground_truth_class = 4
ground_truth_class_name = class_labels[ground_truth_class]

# Check if the prediction is correct
correct_prediction = predicted_class == ground_truth_class
print(f'Ground Truth Class: {ground_truth_class_name}')
print(f'Correct Prediction: {correct_prediction}')