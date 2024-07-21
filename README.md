# Project Name

This project demonstrates object detection using YOLOv5 for detecting persons in images, videos, and RTSP streams.

## Overview

The project utilizes YOLOv5, a state-of-the-art object detection model, to detect persons in various media types:
- **Images**: Detects persons in individual image files.
- **Videos**: Initially designed to process video files, although implementation is commented out.
- **RTSP Streams**: Intended to process RTSP streams for real-time person detection, but currently commented out.

The project focuses on demonstrating how to:
- Load a custom YOLOv5 model trained on person detection.
- Process directories of images to detect and annotate persons.
- Integrate with video streams and RTSP streams for real-time detection and annotation (commented out for now).

## Requirements
-Python 3.x
- torch (PyTorch)
- torchvision
- opencv-python
- Pillow (PIL)
- matplotlib
- ultralytics.yolov5

- I have created a "requirements.txt" fo that you can install all the dependencies related to this project.
- You can simply create a Conda environment and inside the environment install all dependencies like python==3.10.0, ...
- Open the code "count_person_in_frame.p" and uncomment the function which you wanna use and then ren after saving the file.
- The result will be save accordingly in your directory.

## Results
- I have uploaded some of my results also in the reposetory.
