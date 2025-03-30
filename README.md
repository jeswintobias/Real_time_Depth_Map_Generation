# Real_time_Depth_Map_Generation

DEMO VIDEO LINK: https://drive.google.com/file/d/156H4GT8FrMF9fE9XNOp9-EbEGBONGBYc/view?usp=sharing

# Real-Time Depth Estimator

**Turning Pixels into Depth Perception with Computer Vision & Deep Learning!**

## Problem Statement
The challenge of real-time depth estimation lies in analyzing pixel variations to infer the relative distance of objects from a camera. Instead of measuring exact physical distances, our system detects how pixel displacement and intensity changes indicate depth. By leveraging computer vision techniques such as disparity mapping (for stereo vision) and deep learning-based monocular depth estimation, we extract depth information dynamically from standard 2D images.

## Team Members
- **Dhanvanth S** (EC23B1097)
- **Sooraj S** (CS23I1006)
- **Jeswin Tobias** (CS23I1013)

## Why Do This?
Understanding depth without expensive LiDAR sensors is like giving a regular camera superhuman vision—unlocking 3D perception using just pixels! This project harnesses the power of computer vision and deep learning to extract depth information, revolutionizing AI-driven surveillance, robotics, and real-time object interaction. Instead of relying on bulky, costly hardware, we make machines smarter, not harder—helping them interpret depth dynamically, just like the human eye, but faster and more precise.

## Solution Overview
We venture into the world of pixel sorcery, where every shift, distortion, and transformation in an image reveals hidden depth cues. Our system processes video frames in real time, tracking pixel intensities and morphing objects into a depth-aware experience without ever measuring distance directly.

### Methods Used:
- **Monocular Depth Estimation:** Deep learning models predict depth from a single image.
- **Stereo Vision:** Using disparity mapping to compute relative depth.
- **Neural Networks:** Torch-based models infer depth from trained datasets.
- **OpenCV Pipelines:** Efficient real-time video frame processing.

### Applications:
- **Robotics** – Enhancing autonomous navigation with depth-aware perception.
- **AI-Driven Surveillance** – Identifying and tracking objects in 3D space.
- **Augmented Reality (AR)** – Improving real-world interactions with depth estimation.
- **Autonomous Vehicles** – Real-time object detection without costly sensors.

This project pushes the limits of software-driven depth estimation, making advanced 3D vision accessible with just a basic camera!

## Tech Stack
- ✅ **Programming Languages:** Python, HTML, CSS
- ✅ **Libraries:** OpenCV, Torch, NumPy, Matplotlib, Flask
- ✅ **Frameworks:** PyTorch, Torchvision
- ✅ **Models Used:** Depth Anything

## Usage
- **Live Mode:** Use your webcam to visualize real-time depth.
- **Video Processing:** Apply depth estimation to a video file.
- **Model Fine-Tuning:** Train the model on custom datasets for specific use cases.

## Future Enhancements
- 🔹 Improve model accuracy with self-supervised learning.
- 🔹 Optimize performance for low-power edge devices.
- 🔹 Expand support for multi-camera setups.
- 🔹 Integrate 3D scene reconstruction for richer depth visualization.

---

🚀 *This project makes real-time depth estimation more accessible and powerful using just software—no costly hardware required!*
