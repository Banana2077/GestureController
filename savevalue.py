import cv2
import math
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
cap = cv2.VideoCapture(0)
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

def group_hands(multi_hand_landmarks, w, h):
    """
    Groups hands that are close to each other.
    Returns a list of lists of hand_landmarks.
    """
    if not multi_hand_landmarks:
        return []
    
    # If there is only one hand, it's a single group
    if len(multi_hand_landmarks) == 1:
        return [[multi_hand_landmarks[0]]]
        
    # If there are two hands, check proximity
    h1 = multi_hand_landmarks[0]
    h2 = multi_hand_landmarks[1]
    
    p1 = h1.landmark[0]
    p2 = h2.landmark[0]
    palm1 = (p1.x * w, p1.y * h)
    palm2 = (p2.x * w, p2.y * h)
    
    dist = math.sqrt((palm1[0] - palm2[0])**2 + (palm1[1] - palm2[1])**2)
    
    h1_w = (max(lm.x for lm in h1.landmark) - min(lm.x for lm in h1.landmark)) * w
    h1_h = (max(lm.y for lm in h1.landmark) - min(lm.y for lm in h1.landmark)) * h
    size1 = max(h1_w, h1_h)
    
    h2_w = (max(lm.x for lm in h2.landmark) - min(lm.x for lm in h2.landmark)) * w
    h2_h = (max(lm.y for lm in h2.landmark) - min(lm.y for lm in h2.landmark)) * h
    size2 = max(h2_w, h2_h)
    
    avg_size = (size1 + size2) / 2.0
    
    # If palms are within 2.5 times the average hand size, they are "close"
    if dist < (avg_size * 2.5):
        return [[h1, h2]]
    else:
        return [[h1], [h2]]

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

        if results.multi_hand_landmarks:
            hand_groups = group_hands(results.multi_hand_landmarks, w, h)
            
            for hand_group in hand_groups:
                # Combine coordinates of all hands in the group
                x_coords = []
                y_coords = []
                for hand_landmarks in hand_group:
                    x_coords.extend([lm.x * w for lm in hand_landmarks.landmark])
                    y_coords.extend([lm.y * h for lm in hand_landmarks.landmark])
                    
                    # Draw landmarks on the original camera view
                    mp_drawing.draw_landmarks(original, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                hand_w_raw = max(x_coords) - min(x_coords)
                hand_h_raw = max(y_coords) - min(y_coords)
                hand_size = max(hand_w_raw, hand_h_raw)
                
                # Dynamic square crop calculation based on trackbar percentage
                half_crop_size = hand_size * (0.5 + (handScale / 100.0))
                x_center = (min(x_coords) + max(x_coords)) / 2.0
                y_center = (min(y_coords) + max(y_coords)) / 2.0
                
                x_min = int(x_center - half_crop_size)
                x_max = int(x_center + half_crop_size)
                y_min = int(y_center - half_crop_size)
                y_max = int(y_center + half_crop_size)
                
                # Clip to image boundaries
                x_min, x_max = max(x_min, 0), min(x_max, w)
                y_min, y_max = max(y_min, 0), min(y_max, h)
                
                # Check for valid bounding box shape
                if x_max > x_min and y_max > y_min:
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

                        # Paste back to output canvas
                        output[y_min:y_max, x_min:x_max] = hand_filtered
                        
                        # Generate cropped image for saving/testing (640x640)
                        save_img = cv2.resize(hand_filtered, (640, 640), interpolation=cv2.INTER_NEAREST)
                        
                        # Save if save mode is ON
                        if save_crop:
                            try:
                                classI = presskey
                                img_path = f"images/hand{classI}_{frame_id}.jpg"
                                cv2.imwrite(img_path, save_img)
                                
                                crop_w = x_max - x_min
                                crop_h = y_max - y_min
                                
                                x_center_rel = ((min(x_coords) + max(x_coords)) / 2.0 - x_min) / crop_w
                                y_center_rel = ((min(y_coords) + max(y_coords)) / 2.0 - y_min) / crop_h
                                width_rel = (max(x_coords) - min(x_coords)) / crop_w
                                height_rel = (max(y_coords) - min(y_coords)) / crop_h
                                
                                label_path = f"labels/hand{classI}_{frame_id}.txt"
                                with open(label_path, "w") as f:
                                    f.write(f"{classI} {x_center_rel:.6f} {y_center_rel:.6f} {width_rel:.6f} {height_rel:.6f}\n")
                                
                                print(f"Saved cropped hand(s) {frame_id}: Class {classI} (Box: {x_center_rel:.4f}, {y_center_rel:.4f})")
                                frame_id += 1
                            except Exception as e:
                                print(f"Error saving crop: {e}")
                        
                        # Run YOLO detection on cropped hand(s) if detect mode is ON and model loaded
                        if detect and model_loaded and model_shadow is not None:
                            try:
                                results_shadow = model_shadow(save_img, verbose=False)
                                crop_w = x_max - x_min
                                crop_h = y_max - y_min
                                for shadow_result in results_shadow:
                                    shadow_boxes = shadow_result.boxes
                                    if shadow_boxes is not None:
                                        for shadow_box in shadow_boxes:
                                            nx1, ny1, nx2, ny2 = map(float, shadow_box.xyxyn[0].tolist())
                                            # Map back to full frame
                                            x1s = int(x_min + nx1 * crop_w)
                                            y1s = int(y_min + ny1 * crop_h)
                                            x2s = int(x_min + nx2 * crop_w)
                                            y2s = int(y_min + ny2 * crop_h)
                                            conf_s = float(shadow_box.conf[0])
                                            cls_s = int(shadow_box.cls[0])
                                            
                                            # Draw predicted bounding box
                                            cv2.rectangle(output, (x1s, y1s), (x2s, y2s), (0, 255, 0), 2)
                                            label_text = f"{model_shadow.names[cls_s]} {conf_s:.2f}"
                                            cv2.putText(output, label_text, (x1s, y1s - 10), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Shadow detection error: {e}")

                        # Draw encompass blue bounding box on display output per group
                        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

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