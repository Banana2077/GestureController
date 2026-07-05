import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import numpy as np
import os
import time
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# Initialize MediaPipe For Hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0) # 0 for general model, 1 for landscape

# Create directories if they do not exist
os.makedirs("images", exist_ok=True)
os.makedirs("labels", exist_ok=True)

save_crop = False  # Set to True only when you want to save
frame_id = 0  # Increment this after each save
detect = False # use for check it use detection

# YOLO PoseDetection Model Paths
model_path = "C:/Users/comsc/best_project/GestureController/best.pt"

# Load YOLO model
model_shadow = None
model_loaded = False
if os.path.exists(model_path):
    try:
        model_shadow = YOLO(model_path)
        model_loaded = True
        print(f"Successfully loaded YOLO model from {model_path}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
else:
    print(f"Warning: YOLO model not found at '{model_path}'. Running detection will be disabled, but dataset collection will work.")

# Configuration
BACKGROUND_TYPE = "color"  # Change this to "color", "image", or "blur"
BG_COLOR = (0, 0, 0)  # background color in BGR format

# Webcam Initialization (try camera index 1 first, fallback to 0 if it fails)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Warning: Camera index 1 could not be opened. Falling back to camera index 0.")
    cap = cv2.VideoCapture(0)

cv2.namedWindow("Controls", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("Controls", 500, 150)
def nothing(x): pass

# Create adjustment sliders
cv2.createTrackbar("Threshold", "Controls", 70, 255, nothing)
cv2.createTrackbar("handScale", "Controls", 20, 255, nothing)

presskey = 0

def apply_filters(roi, brightness, contrast, saturation, warmth):
    roi = cv2.convertScaleAbs(roi, alpha=contrast / 50.0, beta=brightness - 50)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (saturation / 50.0), 0, 255)
    roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    b, g, r = cv2.split(roi)
    r = np.clip(r + (warmth - 50), 0, 255).astype(np.uint8)
    roi = cv2.merge((b, g, r))
    return roi

print("--- Start YOLO Hand Shadow Dataset Collector ---")
print("Press 'a' to toggle Save Mode (ON/OFF)")
print("Press 'd' to toggle YOLO Shadow Detection (requires model)")
print("Press '0'-'9' to change the class ID for saving")
print("Press 'q' to quit")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        original = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultsBg = selfie_segmentation.process(rgb_frame)
        condition = np.stack((resultsBg.segmentation_mask,) * 3, axis=-1) > 0.1 # Threshold can be adjusted

        if BACKGROUND_TYPE == "color":
            background = np.zeros(frame.shape, dtype=np.uint8)
            background[:] = BG_COLOR

        output_frame = np.where(condition, frame, background)
        
        # Get slider values
        thresh = cv2.getTrackbarPos("Threshold", "Controls")
        handScale = cv2.getTrackbarPos("handScale", "Controls")
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Output contains only the hand silhouette inside the bounding box(es)
        output = np.zeros_like(output_frame)

        # List to store bounding boxes of detected hands
        PosXmin = []
        PosXmax = []
        PosYmin = []
        PosYmax = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min = int(min(x_coords)) - handScale
                x_max = int(max(x_coords)) + handScale
                y_min = int(min(y_coords)) - handScale
                y_max = int(max(y_coords)) + handScale
                
                # Clip to image boundaries
                x_min, x_max = max(x_min, 0), min(x_max, w)
                y_min, y_max = max(y_min, 0), min(y_max, h)
                
                # Check for valid bounding box shape
                if x_max > x_min and y_max > y_min:
                    PosXmin.append(x_min)
                    PosXmax.append(x_max)
                    PosYmin.append(y_min)
                    PosYmax.append(y_max)
                    
                    hand_roi = output_frame[y_min:y_max, x_min:x_max]
                    
                    if hand_roi.size > 0:
                        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

                        hand_canvas = np.zeros_like(hand_roi)
                        hand_canvas[:] = (255, 255, 255)
                        mask_3ch = cv2.merge([mask, mask, mask])
                        hand_result = np.where(mask_3ch == 255, hand_canvas, 0)

                        # Apply filters to hand only
                        hand_filtered = apply_filters(hand_result, 50, 50, 50, 50)

                        # Resize/paste back to output canvas
                        output[y_min:y_max, x_min:x_max] = hand_filtered

                # Draw landmarks on the original camera view
                mp_drawing.draw_landmarks(original, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Create a clean copy of the output (only black background + white hands) for saving
        # so that blue bounding boxes, green YOLO boxes, or text are NOT saved as training data.
        save_frame = output.copy()

        # Run YOLO detection on 'output' if detect mode is ON and model is loaded
        if detect and model_loaded and model_shadow is not None:
            try:
                results_shadow = model_shadow(output, show=False)
                for shadow_result in results_shadow:
                    shadow_boxes = shadow_result.boxes
                    if shadow_boxes is not None:
                        for shadow_box in shadow_boxes:
                            x1s, y1s, x2s, y2s = map(int, shadow_box.xyxy[0].tolist())
                            conf_s = float(shadow_box.conf[0])
                            cls_s = int(shadow_box.cls[0])
                            
                            # Draw predicted bounding box
                            cv2.rectangle(output, (x1s, y1s), (x2s, y2s), (0, 255, 0), 2)
                            
                            # Draw prediction label
                            label_text = f"{model_shadow.names[cls_s]} {conf_s:.2f}"
                            cv2.putText(output, label_text, (x1s, y1s - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"Shadow detection error: {e}")

        # If hands are detected, draw the main encompassing bounding box and handle saving
        if len(PosXmin) > 0:
            MiNx = min(PosXmin)
            MaXx = max(PosXmax)
            MiNy = min(PosYmin)
            MaXy = max(PosYmax)
            
            classI = presskey  # Class index

            # Save frame and label
            if save_crop:
                try:
                    if save_frame.size > 0:
                        img_path = f"images/hand{classI}_{frame_id}.jpg"
                        cv2.imwrite(img_path, save_frame)

                        # Calculate YOLOv8 label (normalized coordinates)
                        img_h, img_w = save_frame.shape[:2]
                        x_center = ((MiNx + MaXx) / 2.0) / img_w
                        y_center = ((MiNy + MaXy) / 2.0) / img_h
                        width = (MaXx - MiNx) / img_w
                        height = (MaXy - MiNy) / img_h

                        label_path = f"labels/hand{classI}_{frame_id}.txt"
                        with open(label_path, "w") as f:
                            f.write(f"{classI} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                        print(f"Saved frame {frame_id}: Class {classI} (Box: {x_center:.4f}, {y_center:.4f}, {width:.4f}, {height:.4f})")
                        frame_id += 1
                except Exception as e:
                    print(f"Error saving crop: {e}")

            # Draw encompass blue bounding box on display output only (not in saved file)
            cv2.rectangle(output, (int(MiNx), int(MiNy)), (int(MaXx), int(MaXy)), (255, 0, 0), 2)

    except Exception as e:
        print(f"Main loop error: {e}")

    # Add UI overlays on 'original' window
    cv2.putText(original, f'Save Mode: {"ON" if save_crop else "OFF"}', (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if save_crop else (0, 0, 255), 2)
    cv2.putText(original, f'Save Class: {presskey}', (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(original, f'Detect Mode: {"ON" if detect else "OFF"}', (30, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0) if detect else (100, 100, 100), 2)

    cv2.imshow("Original", original)
    cv2.imshow('Webcam Background Removal', output_frame)
    cv2.imshow("Filtered Hand", output)

    key = cv2.waitKey(1) & 0xFF
    if 48 <= key <= 57:  # 0 ~ 9 key
        presskey = key - 48
        print(f"Switched save class to: {presskey}")
        
    if key == ord('a'):  # Toggle save mode
        save_crop = not save_crop
        print(f"Save mode: {'ON' if save_crop else 'OFF'}")
    elif key == ord('d'):  # Toggle detection mode
        if not model_loaded:
            print("Cannot enable detection: YOLO model was not loaded successfully.")
        else:
            detect = not detect
            print(f"Detection mode: {'ON' if detect else 'OFF'}")
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
print("Finished!")