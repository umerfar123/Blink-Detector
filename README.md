# Blink Detector 👀

A Python-based software application that employs computer vision algorithms to accurately identify and quantify eye blinks from real-time or recorded video footage. 
The application can be utilized for various purposes, such as fatigue detection, driver alertness monitoring, or human-computer interaction.

## Introduction
This Python application leverages OpenCV and MediaPipe to accurately detect and analyze eye blinks in real-time or from recorded video footage. 
By utilizing facial landmark detection and eye aspect ratio (EAR) calculations, the application provides a reliable and efficient solution for various applications, including:

* Fatigue Detection: Monitoring driver alertness or employee productivity.
* Human-Computer Interaction: Enabling hands-free control of devices.
* Medical Research: Studying eye movement patterns and disorders.

### Media Pipe Face Mesh
MediaPipe Face Mesh Integration
This project utilizes MediaPipe Face Mesh, a powerful machine learning solution for high-accuracy face landmark detection. 
MediaPipe provides a pre-trained model that can detect and track hundreds of facial landmarks in real-time. To know more identified points visit this github [Repo.](https://github.com/HotaruK/mediapipe_demo/blob/main/keypoints/face_mesh_no.jpg)

## How It Works

* **Video Acquisition**: The application captures video frames from the specified source, which can be a webcam, a pre-recorded video file, or a live stream.
* **Face Mesh Detection**: MediaPipe Face Mesh is employed to detect and track the facial landmarks within each frame. This includes identifying key points around the eyes,
such as the corners, upper and lower eyelids, and inner eye corners.
* **Eye Landmark Extraction**: Once the face mesh is detected, the system extracts the specific landmark points corresponding to the eyes. These points are essential for calculating the eye aspect ratio (EAR).
* **EAR Calculation**: The EAR is computed using the extracted eye landmarks. It measures the ratio of the horizontal distance between the outer eye corners to the vertical distance between the upper and lower eyelid points.

![Capture](https://github.com/user-attachments/assets/37a71322-0e5a-44b2-be99-105cc68072ff)

* **Blink Detection**: A threshold value is defined for the EAR. When the calculated EAR falls below this threshold, it indicates a closed eye, suggesting a blink.

## Customization

You can customize the application's behavior by adjusting the following parameters:

* EAR threshold: The threshold value used to determine if a blink has occurred.
* Video source: The path to the video file to be processed.
* Display options: Control whether to display the video frames and detected blinks.

## Usage

1. Clone this repository using :

   ```
   git clone 
   ```
2. Install mediapipe and opencv libraries :

   ```python
    pip install opencv-python
   ```
    ```python
    pip install mediapipe
   ```
3. Run the main file
   
   ```python
    python main.py
    ```
___
> [!NOTE]
> To Exit the Window press 'x'
    

# Demo Videos

