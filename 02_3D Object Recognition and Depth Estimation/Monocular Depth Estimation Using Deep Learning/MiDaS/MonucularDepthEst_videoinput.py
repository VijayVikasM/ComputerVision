import cv2
import torch
import time
import numpy as np


#Load a MiDas model for depth estimation
#model_type = "DPT_Large"   #MiDaS v3 - Large (highest accuracy, slowest inferencce speed)
#model_type = "DPT_Hybrid"   #MiDaS v3 - Hybrid (medium accuracy, medium inferencce speed)
model_type = "MiDaS_small"   #MiDaS v2.1 - Small (highest accuracy, highest inferencce speed)


# Repository path and model loading
try:
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
except Exception as e:
    print("Error loading model:", e)
    exit()

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms based on model type
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Open up the video capture from a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    start = time.time()

    # Convert and process the image for depth estimation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # Depth estimation and processing
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Normalize depth map for visualization
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        end = time.time()
        fps = 1 / (end - start)

        # Display FPS and depth map
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth Map', depth_map)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
