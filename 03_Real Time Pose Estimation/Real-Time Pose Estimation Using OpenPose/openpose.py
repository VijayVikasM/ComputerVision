import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle_with_horizontal(point1, point2):
    # Calculate the angle of the line formed by point1 and point2 relative to the horizontal
    a = np.array(point1)  # Right hip
    b = np.array(point2)  # Right ankle (approximates the shoe position)
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])  # Calculate angle from horizontal
    angle = np.abs(radians * 180.0 / np.pi)  # Convert to degrees
    return angle

# Open video file or camera
cap = cv2.VideoCapture("mjtilt.mp4")  # 
# video source: https://www.youtube.com/@michaeljacksonhd9516

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Pose
    results = pose.process(rgb_frame)

    # If poses are detected
    if results.pose_landmarks:
        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate tilt angle using the angle of the line between right hip and right ankle
        tilt_angle = calculate_angle_with_horizontal(right_hip, right_ankle)

        # Draw landmarks and connections on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display tilt angle on the frame
        cv2.putText(frame, f'Tilt Angle: {int(tilt_angle)} degrees', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame with calculated tilt angle
    cv2.imshow("Michael Jackson Tilt Angle", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
