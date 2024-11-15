# Computer Vision Projects

This repository contains a collection of computer vision projects aimed at building a portfolio for a role in humanoid robotics. Each section covers different essential skills and techniques relevant to computer vision, such as object detection, tracking, depth estimation and more.

---

## 1. Object Detection and Tracking in Dynamic Environments

### Project 1: Multi-Object Detection and Tracking in Video Feeds
- **Description**: Detect and track multiple objects in real-time video feeds (e.g., people, vehicles, animals) using YOLO or Faster R-CNN and the SORT algorithm.
- **Skills**: YOLO, SORT, OpenCV.
- **Datasets**: [Camera feed Roadside](https://github.com/intel-iot-devkit/sample-videos?tab=readme-ov-file).

### Project 2: Motion-Based Object Tracking with Background Subtraction
- **Description**: Identify and track moving objects using background subtraction algorithms like MOG2 or KNN for dynamic backgrounds.
- **Skills**: Background subtraction, OpenCV.
- **Datasets**: [Human Tracking dataset](http://www.santhoshsunderrajan.com/datasets.html).

---

## 2. 3D Object Recognition and Depth Estimation

### Project 1: Depth Estimation from Stereo Images
- **Description**: Estimate depth from stereo images by implementing disparity mapping techniques.
- **Skills**: Stereo vision, depth estimation, OpenCV.
- **Datasets**: [Stereo Image Pairs](https://vision.middlebury.edu/stereo/data/scenes2021/).

### Project 2: Monocular Depth Estimation Using Deep Learning on a live camera feed
- **Description**: Use a deep learning model (like MiDaS and DepthAnything ) to predict relative depth from a live web cam feed.
- **Skills**: CNNs, Vision Transformers, monocular depth estimation, PyTorch
- **Datasets**: [NYU Depth V2 dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

---

## 3. Pose Estimation for Human-Robot Interaction

### Project 1: Real-Time Pose Estimation Using OpenPose
- **Description**: Detect and analyze human poses in real-time video feeds using OpenPose.
- **Skills**: Pose estimation, OpenPose, OpenCV.
- **Datasets**: [COCO keypoints dataset](http://cocodataset.org/#keypoints-2019).

### Project 2: Gesture Recognition from Human Poses
- **Description**: Recognize basic gestures (like waving, thumbs up) using keypoints from pose estimation.
- **Skills**: Pose keypoints, gesture recognition, OpenCV.
- **Datasets**: [NTU RGB+D dataset](http://rose1.ntu.edu.sg/datasets/actionRecognition.asp).

---

## 4. Visual SLAM (Simultaneous Localization and Mapping)

### Project 1: 2D Feature-Based SLAM Using ORB
- **Description**: Implement a feature-based SLAM system using ORB for feature detection and matching.
- **Skills**: ORB, feature matching, OpenCV.
- **Datasets**: [TUM RGB-D dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset).

### Project 2: Visual Odometry with Monocular Camera
- **Description**: Track a camera’s movement using visual odometry with feature matching techniques.
- **Skills**: Feature matching, visual odometry, OpenCV.
- **Datasets**: [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

---

## 5. Face Recognition and Emotion Detection

### Project 1: Real-Time Face Recognition with OpenCV and dlib
- **Description**: Implement real-time face recognition using dlib’s face recognition model.
- **Skills**: Face recognition, dlib, OpenCV.
- **Datasets**: [LFW dataset](http://vis-www.cs.umass.edu/lfw/).

### Project 2: Emotion Detection Using a Convolutional Neural Network
- **Description**: Classify facial emotions like happiness, sadness, and anger using a CNN.
- **Skills**: CNN, emotion classification, PyTorch/TensorFlow.
- **Datasets**: [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

---

## 6. Gesture-Based Control System

### Project 1: Hand Gesture Recognition Using MediaPipe
- **Description**: Recognize gestures like “peace” and “thumbs up” using MediaPipe’s hand-tracking model.
- **Skills**: MediaPipe, OpenCV.
- **Datasets**: Custom hand gesture data with MediaPipe.

### Project 2: Sign Language Recognition Using Keypoints
- **Description**: Recognize American Sign Language (ASL) alphabet gestures using a hand keypoint model.
- **Skills**: Keypoint detection, hand gesture classification.
- **Datasets**: [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

---

## 7. Grasp Detection and Manipulation

### Project 1: Grasp Detection with Object Detection Models
- **Description**: Detect object boundaries and predict grasp points with bounding boxes and orientation.
- **Skills**: Object detection, grasp point estimation, OpenCV.
- **Datasets**: [Cornell Grasping dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php).

### Project 2: Simulated Grasp Planning with Reinforcement Learning
- **Description**: Simulate a reinforcement learning model to "grasp" objects in a virtual environment using OpenAI Gym or PyBullet.
- **Skills**: Reinforcement learning, PyBullet.
- **Datasets**: Simulated synthetic images.

---

## 8. Scene Understanding and Semantic Segmentation

### Project 1: Semantic Segmentation Using DeepLabV3
- **Description**: Perform semantic segmentation on scenes to identify objects like furniture and people using DeepLabV3.
- **Skills**: Semantic segmentation, DeepLabV3.
- **Datasets**: [Cityscapes dataset](https://www.cityscapes-dataset.com/) or [ADE20K dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/).

### Project 2: Indoor Scene Parsing with Mask R-CNN
- **Description**: Use Mask R-CNN for instance segmentation in indoor scenes, labeling distinct objects and instances.
- **Skills**: Mask R-CNN, instance segmentation.
- **Datasets**: [SUN RGB-D dataset](https://rgbd.cs.princeton.edu/).

---

## 9. Anomaly Detection in Real-Time Vision Feeds

### Project 1: Unusual Activity Detection in Videos Using Autoencoders
- **Description**: Detect anomalies in video frames by training an autoencoder on normal activities and flagging high-error frames.
- **Skills**: Autoencoders, anomaly detection.
- **Datasets**: [UCSD Anomaly Detection dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm).

### Project 2: Real-Time Fall Detection Using Pose Estimation
- **Description**: Use pose estimation to detect falls by analyzing angles and keypoints in body posture.
- **Skills**: Pose estimation, keypoint analysis.
- **Datasets**: [Charades dataset](http://allenai.org/plato/charades/).

---

## 10. Object Recognition for Task Automation

### Project 1: Household Object Recognition Using CNNs
- **Description**: Train a CNN to recognize common household objects like books, cups, and remote controls.
- **Skills**: Object recognition, CNN.
- **Datasets**: [COCO dataset](https://cocodataset.org/) or [ImageNet dataset](http://www.image-net.org/).

### Project 2: Scene-Specific Object Detection for Kitchen Items
- **Description**: Detect kitchen items (e.g., plates, utensils) in images for potential automation tasks.
- **Skills**: Object detection, transfer learning.
- **Datasets**: Custom dataset or COCO subsets.

---



## References

### Motion-Based Object Tracking with Background Subtraction**
**example_video.avi**

```bibtex
@inproceedings{sunderrajan2015robust,
  title={Robust Multiple Camera Tracking with Spatial And Appearance Contexts},
  author={Sunderrajan, Santhoshkumar and Jagadeesh, Vignesh and Manjunath, BS},
  year={2015},
}

@inproceedings{sunderrajan2013Multiple, 
  author={Sunderrajan, S. and Manjunath, B.S.}, 
  booktitle={Distributed Smart Cameras (ICDSC), 2013 Seventh International Conference on}, 
  title={Multiple view discriminative appearance modeling with IMCMC for distributed tracking}, 
  year={2013}, 
  month={Oct}, 
  pages={1-7}, 
  doi={10.1109/ICDSC.2013.6778203},
}

@inproceedings{ni2010distributed,
  title={Distributed particle filter tracking with online multiple instance learning in a camera sensor network},
  author={Ni, Zefeng and Sunderrajan, Santhoshkumar and Rahimi, Amir and Manjunath, BS},
  booktitle={Image Processing (ICIP), 2010 17th IEEE International Conference on},
  pages={37--40},
  year={2010},
  organization={IEEE}
}

