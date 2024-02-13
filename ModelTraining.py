import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split

# Dataset folder paths
dataset_folders = [
    r"C:\Users\vicky\OneDrive\Desktop\Images_for_db\gaussian_filtered_images\gaussian_filtered_images\Mild",
    r"C:\Users\vicky\OneDrive\Desktop\Images_for_db\gaussian_filtered_images\gaussian_filtered_images\Severe",
    r"C:\Users\vicky\OneDrive\Desktop\Images_for_db\gaussian_filtered_images\gaussian_filtered_images\Moderate",
    r"C:\Users\vicky\OneDrive\Desktop\Images_for_db\gaussian_filtered_images\gaussian_filtered_images\Proliferate_DR",
    r"C:\Users\vicky\OneDrive\Desktop\Images_for_db\gaussian_filtered_images\gaussian_filtered_images\No_DR"
]

# Initialize lists to store preprocessed images and labels
preprocessed_images = []
labels = []

# Data preprocessing parameters
target_size = (180, 200)  # Desired image size
normalize_pixels = True
convert_to_grayscale = True

# Load images and assign labels
for label, folder_path in enumerate(dataset_folders):
    files = os.listdir(folder_path)
    for file in files:
        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path)

        # Resize the image
        image = image.resize(target_size)

        # Grayscale the image
        if convert_to_grayscale:
            image = image.convert('L')

        # Normalize pixel values to [0, 1]
        if normalize_pixels:
            image = np.array(image) / 255.0

        # Append the preprocessed image and label to the lists
        preprocessed_images.append(image)
        labels.append(label)

# Convert the lists to NumPy arrays
X_train = np.array(preprocessed_images)
y_train = np.array(labels)

X_train = X_train.reshape(-1, 180, 200, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the model
# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(180, 200, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))  # Increased filters
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))  # Increased filters
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Increased units
model.add(Dense(len(dataset_folders), activation='softmax'))

# Summary of model architecture
model.summary()

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
model.save(r'C:\Users\PC-ACER\PycharmProjects\pythonProject\ModelTraining.h5')
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
print("Evaluation complete")

# Assuming X_test contains your test data
# Get the predicted probabilities for each class
y_score = model.predict(X_test)

# Assuming y_test is a 1D array of class labels
# Convert it to a 2D array of one-hot encoded labels
y_test_one_hot = label_binarize(y_test, classes=np.arange(len(dataset_folders)))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(dataset_folders)):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(dataset_folders)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"], label='Micro-average (AUC = {0:0.2f})'.format(roc_auc["micro"]))

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()