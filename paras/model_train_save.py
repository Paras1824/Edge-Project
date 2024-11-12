import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import shutil

# Step 1: Set paths to the dataset
dataset_path = "/home/user/Desktop/pproject/paras/dataset"
  # Replace with your actual path
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")
validation_dir = os.path.join(dataset_path, "validation")

# Step 2: Split test data into validation and test sets if validation folder doesn't exist
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)
    for class_name in os.listdir(test_dir):
        class_test_path = os.path.join(test_dir, class_name)
        class_validation_path = os.path.join(validation_dir, class_name)
        os.makedirs(class_validation_path, exist_ok=True)
        
        # Split images into 50% validation and 50% test
        images = os.listdir(class_test_path)
        validation_images, test_images = train_test_split(images, test_size=0.5, random_state=42)
        
        # Move validation images
        for image in validation_images:
            shutil.move(os.path.join(class_test_path, image), os.path.join(class_validation_path, image))

# Step 3: Preprocess the data using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

# Step 4: Build the CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# Step 5: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Step 7: Save the trained model
model.save('emotion_detection_model.h5')

# Step 8: Load the model for inference
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Step 9: Function to predict emotion from a single image
def predict_emotion(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))  # Resize to match model input
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (grayscale)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    
    # Predict emotion
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Map predicted class to emotion label
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    return emotions[predicted_class[0]]

# Step 10: Test the model with a sample image
test_image_path = 'path_to_test_image.jpg'  # Replace with actual test image path
predicted_emotion = predict_emotion(test_image_path)
print("Predicted emotion:", predicted_emotion)
