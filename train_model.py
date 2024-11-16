import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define directories (replace with the path to your dataset)
data_dir = '/Users/Arron/Documents/AppCon/fire-dataset'

# Use ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load the dataset with ImageDataGenerator
train_generator = datagen.flow_from_directory(
    data_dir, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='binary', 
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='binary', 
    subset='validation'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification (fire or no fire)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save('fire_detection_model.h5')
