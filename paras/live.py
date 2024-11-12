import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Define the emotion labels
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Function to preprocess the frame
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, (48, 48))  # Resize to match model input
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    processed_frame = np.expand_dims(normalized_frame, axis=-1)  # Add channel dimension
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
    return processed_frame

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or change for other cameras

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        print("Failed to capture video frame. Exiting...")
        break
    
    # Process the frame for emotion detection
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_class = np.argmax(prediction, axis=1)[0]
    emotion_label = emotions[predicted_class]
    
    # Display the prediction on the video feed
    cv2.putText(frame, f"Emotion: {emotion_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the video feed with the emotion label
    cv2.imshow("Live Emotion Detection", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
