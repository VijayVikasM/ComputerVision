import torch
import cv2
import numpy as np

# Load MiDaS model for depth estimation
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.small_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Load YOLOv5 model for object detection
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(device)

# Open the camera or video stream (change "0" to video file path if needed)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Perform object detection with YOLOv5
    results = yolo(frame)
    detections = results.xyxy[0].cpu().numpy()  # bounding boxes, confidence, class

    # Perform depth estimation with MiDaS
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        depth_map = midas(input_batch).squeeze().cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        # Resize the depth map to match the frame's dimensions
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))


    # Loop over detections and calculate 3D position
    for x1, y1, x2, y2, conf, cls in detections:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # bounding box coordinates
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # center of the box

        # Estimate the object's depth by averaging the depth map values within the bounding box
        object_depth = np.mean(depth_map[y1:y2, x1:x2])

        # Convert to approximate 3D coordinates
        # Assuming depth represents distance from camera in arbitrary units
        target_coordinates = (center_x, center_y, object_depth)
        
        # Draw bounding box and display depth
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Depth: {object_depth:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Object ID: {int(cls)}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the estimated 3D coordinates in the frame
        print(f"3D Coordinates for Object {int(cls)}: {target_coordinates}")

    # Show the frame with depth information
    depth_map_vis = (depth_map * 255).astype(np.uint8)
    depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_MAGMA)
    cv2.imshow("Depth Map", depth_map_vis)
    cv2.imshow("Object Detection and Depth Estimation", frame)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
