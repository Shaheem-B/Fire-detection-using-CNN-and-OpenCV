A fire detection system combines Convolutional Neural Network and OpenCV for accurate and real-time detection.

🔹 Role of CNN:
     Learns fire features (color, texture, shape) from data.
     Provides high accuracy through deep learning.
     Classifies images as fire or no fire after training on large datasets.

🔹 Role of OpenCV:
     Captures and preprocesses video frames (color filtering, noise reduction).
     Detects fire based on color and motion but is less reliable in varying conditions.
     Passes frames to CNN for classification and triggers alerts if fire is detected.

🔹 Key Difference:
     CNN is data-driven (higher accuracy but slower).
     OpenCV is rule-based (faster but less adaptive).

Together, CNN + OpenCV create an efficient, automated fire detection system with real-time processing and high reliability.
