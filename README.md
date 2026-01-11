Tactical Search and Rescue (SAR) Survivor Detection System
An AI-powered emergency response system designed to detect survivors in disaster zones using computer vision, multi-spectral imaging simulation, and automated emergency alerting.

 Overview
This project leverages the YOLOv8-Pose model to identify human presence and posture in real-time video streams. It is designed for deployment on drones or remote monitoring stations to assist rescue teams in locating survivors by providing visual enhancements and automated location-based alerts.

 Key Features
Human Pose Estimation: Uses YOLOv8-pose to detect human skeletons even in complex environments.

Triple-View Vision System:

Normal View: Standard RGB processing.

Night Vision: Green-spectrum enhancement for low-light simulation.

Thermal Imaging: Heat-map visualization (Inferno colormap) with simulated body temperature readings.

Automated Emergency Alerts: Integrates with Twilio API to send SMS alerts and automated voice calls when survivors are detected.

Real-time HUD: Displays survivor count and precise GPS coordinates (Navsari, Gujarat region) on the video feed.

Optimized Performance: Frame-skipping logic and resized multi-stack display for smooth processing.

 Tech Stack
Language: Python

Libraries: OpenCV, NumPy, Ultralytics (YOLOv8)

Communication: Twilio API

Hardware Compatibility: IP Cameras, Webcams, ESP32-CAM

 Getting Started
1. Prerequisites
Ensure you have Python installed, then install the required dependencies:

Bash

pip install opencv-python numpy ultralytics twilio
2. Twilio Configuration
To enable SMS and Voice alerts, set up your Twilio credentials as environment variables or replace them in the script:

TWILIO_SID

TWILIO_AUTH

TWILIO_PHONE

TARGET_PHONE

3. Running the System
Update the url variable in the script with your IP camera stream or use 0 for a local webcam.

Run the script:

Bash

python sar_detection.py
 System Architecture
The system processes frames through a pipeline that converts raw input into three distinct visual modes while simultaneously running inference to detect human keypoints. If a person is detected, the emergency protocol is triggered.
