# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import pickle
import socket
import threading
import math
import time

# ============================================
# PD MODE (Finger-Curl Tracking)
# ============================================
from pd_mode import PDMode
pd_mode_instance = PDMode()   # Initialize PD Mode (Finger-Curl Tracking, no calibration needed)

# ============================================
# LOAD MODEL
# ============================================
import os
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model_path = "best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f"[YOLO] Loaded model from {model_path}")
else:
    print(f"[YOLO ERROR] Model not found at '{model_path}'")
    model = None

# Mapping dictionary from YOLO class name to Unity gesture name
YOLO_GESTURE_MAPPING = {
    "class_0": "rabbit",
    "class_1": "dog",
    "class_2": "bird",
    "class_3": "cow",
    "class_4": "deer",
}

# ============================================
# MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Selfie Segmentation initialization removed to optimize performance
selfie_segmentation = None

# ============================================
# CAMERA
# ============================================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ============================================
# GESTURE SETTINGS
# ============================================
GESTURE_LABELS = {
    "rabbit": "rabbit",
    "dog": "dog",
    "bird": "bird",
    "cow": "cow",
    "deer": "deer",
    

}

CONFIDENCE_THRESHOLD = 0.5

# ============================================
# MODE
# ============================================
# Modes: "CONTROL", "GESTURE", "PD"
MODE = "CONTROL"

# ============================================
# STATE
# ============================================
HOLD_FRAMES = 0
HOLD_REQUIRED = 5

gesture_text = ""
final_gesture = "dont"

#GESTURE_DURATION = 0.75
GESTURE_DURATION = 3
gesture_start_time = 0

gesture_votes = {}

# ============================================
# JOYSTICK
# ============================================
JOYSTICK_RADIUS = 120  # Fit nicely in 640x480 screen
DEADZONE = 40

fixed_joystick_center = (160, 240)  # Center of the left half of the screen
dynamic_joystick_center = None
joystick_active = False

# Exponential Moving Average (EMA) variables for smoothing left hand coordinates
smoothed_cx = None
smoothed_cy = None
EMA_ALPHA = 0.25  # Smoothing factor (lower = smoother, higher = more responsive)

# ============================================
# TCP (GESTURE)
# ============================================
TCP_HOST = "localhost"
TCP_PORT = 5005

client_conn = None
client_lock = threading.Lock()

running = True

# ============================================
# YOLO THREAD FOR ASYNC INFERENCE
# ============================================
import queue
yolo_queue = queue.Queue(maxsize=1)
yolo_result = ("Not sure...", 0.0, None)
yolo_lock = threading.Lock()

def yolo_worker():
    global yolo_result
    while running:
        try:
            item = yolo_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        if item is None:
            break
            
        img, x_min, y_min, crop_w, crop_h = item
        
        if model is not None:
            try:
                yolo_results = model(img, verbose=False)
                best_conf = 0.0
                best_pred = None
                best_box_rel = None

                for r in yolo_results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            if conf > best_conf:
                                best_conf = conf
                                best_pred = r.names[cls_id]
                                best_box_rel = list(map(float, box.xyxyn[0].tolist()))
                
                with yolo_lock:
                    if best_conf >= CONFIDENCE_THRESHOLD and best_pred is not None:
                        pred_gesture = YOLO_GESTURE_MAPPING.get(best_pred, best_pred)
                        
                        # Map relative coordinates back to full frame
                        nx1, ny1, nx2, ny2 = best_box_rel
                        x1s = int(x_min + nx1 * crop_w)
                        y1s = int(y_min + ny1 * crop_h)
                        x2s = int(x_min + nx2 * crop_w)
                        y2s = int(y_min + ny2 * crop_h)
                        best_box_full = [x1s, y1s, x2s, y2s]
                        
                        yolo_result = (pred_gesture, best_conf, best_box_full)
                        gesture_votes[pred_gesture] = gesture_votes.get(pred_gesture, 0) + 1
                    else:
                        yolo_result = ("Not sure...", 0.0, None)
            except Exception as e:
                print(f"[YOLO Thread Error] {e}")
        yolo_queue.task_done()

yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
yolo_thread.start()

last_tcp_message = ""

# ============================================
# UNITY -> PYTHON RECEIVE
# ============================================
# Timestamp after which GESTURE mode is allowed again (0 = always allowed)
gesture_cooldown_until = 0.0
GESTURE_COOLDOWN_SECS = 3.0

def unity_receiver(conn):
    """
    Background thread: reads newline-delimited commands sent BY Unity.
    Supported commands:
        startminigame  ->  switch to PD mode
        stopminigame   ->  switch to CONTROL + block GESTURE for 3s
    """
    global MODE, HOLD_FRAMES, gesture_cooldown_until
    buf = b""
    try:
        while running:
            chunk = conn.recv(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.decode("utf-8", errors="ignore").strip()
                if not cmd:
                    continue
                print(f"[TCP RECV] Unity said: '{cmd}'")
                if cmd == "startminigame":
                    MODE = "PD"
                    HOLD_FRAMES = 0
                    pd_mode_instance.reset_calibration() # Reset finger smoothing values
                    print("[MODE] -> PD  (startminigame from Unity)")
                elif cmd == "stopminigame":
                    MODE = "CONTROL"
                    HOLD_FRAMES = 0
                    gesture_cooldown_until = time.time() + GESTURE_COOLDOWN_SECS
                    joystick_active = False
                    print(f"[MODE] -> CONTROL  (stopminigame from Unity, GESTURE locked {GESTURE_COOLDOWN_SECS}s)")
    except Exception as e:
        print(f"[TCP RECV ERROR] {e}")

# ============================================
# TCP SERVER
# ============================================
def socket_server():
    global client_conn

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((TCP_HOST, TCP_PORT))
    server.listen(1)

    print(f"[TCP] Waiting for Unity at {TCP_HOST}:{TCP_PORT} ...")

    while running:

        try:
            conn, addr = server.accept()

            with client_lock:
                client_conn = conn

            print(f"[TCP] Unity connected: {addr}")

            # Start a receiver thread for this connection
            recv_thread = threading.Thread(
                target=unity_receiver, args=(conn,), daemon=True
            )
            recv_thread.start()

        except Exception as e:
            print("[TCP SERVER ERROR]", e)

server_thread = threading.Thread(
    target=socket_server,
    daemon=True
)

server_thread.start()

# ============================================
# SEND TCP
# ============================================
def send_tcp(status="", gesture=""):

    global client_conn
    global last_tcp_message

    if gesture:
        message = f"2:{gesture}"

    elif status in ["control", "gesture"]:
        message = f"0:{status}"

    else:
        message = f"1:{status}"

    # ป้องกันส่งซ้ำ
    if message == last_tcp_message:
        return

    last_tcp_message = message

    with client_lock:

        if client_conn is None:
            return

        try:
            client_conn.sendall((message + "\n").encode("utf-8"))

            print(f"[TCP] Sent: {message}")

        except Exception as e:

            print("[TCP ERROR]", e)

            try:
                client_conn.close()
            except:
                pass

            client_conn = None

# ============================================
# UDP (CONTROL)
# ============================================
UDP_HOST = "127.0.0.1"
UDP_PORT = 5006

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

last_udp_message = ""

# ============================================
# SEND UDP
# ============================================
def send_udp(left_state, right_state):

    global last_udp_message

    message = f"L:{left_state}\nR:{right_state}"

    # กัน spam
    if message == last_udp_message:
        return

    last_udp_message = message

    udp_sock.sendto(
        message.encode(),
        (UDP_HOST, UDP_PORT)
    )

# ============================================
# HELPERS
# ============================================
def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

def is_open_hand(hand_landmarks):

    tips = [8, 12, 16, 20]
    mids = [6, 10, 14, 18]

    count = 0

    for tip, mid in zip(tips, mids):

        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mid].y:
            count += 1

    return count >= 4

def finger_states(hand_landmarks, palm_center, radius, w, h):

    states = []

    finger_ids = [4, 8, 12, 16, 20]

    for i in finger_ids:

        tip = hand_landmarks.landmark[i]

        x = int(tip.x * w)
        y = int(tip.y * h)

        d = distance((x, y), palm_center)

        states.append(
            0 if d < radius else 1
        )

    return states

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

# ============================================
# MAIN LOOP
# ============================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    h, w, _ = frame.shape
    
    # Initialize a black silhouette frame for GESTURE mode
    import numpy as np
    silhouette_frame = np.zeros_like(frame)

    left_state = "Idle"
    right_state = "NONE"

    # ============================================
    # DETECT HAND
    # ============================================
    if results.multi_hand_landmarks:

        all_open = len(results.multi_hand_landmarks) == 2 and all(
            is_open_hand(hl)
            for hl in results.multi_hand_landmarks
        )

        # ============================================
        # SWITCH MODE
        # ============================================
        if all_open and MODE == "CONTROL":

            HOLD_FRAMES += 1

            progress = int(
                (HOLD_FRAMES / HOLD_REQUIRED) * 200
            )

            cv2.rectangle(
                frame,
                (10, h - 40),
                (210, h - 15),
                (50, 50, 50),
                -1
            )

            cv2.rectangle(
                frame,
                (10, h - 40),
                (10 + progress, h - 15),
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                "Hold to switch GESTURE mode...",
                (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            if HOLD_FRAMES >= HOLD_REQUIRED:

                # Check 3-second cooldown after stopminigame
                if time.time() < gesture_cooldown_until:
                    remaining_cd = gesture_cooldown_until - time.time()
                    cv2.putText(frame,
                                f"Gesture locked ({remaining_cd:.1f}s)",
                                (10, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 200), 2)
                    HOLD_FRAMES = 0

                else:
                    MODE = "GESTURE"

                    HOLD_FRAMES = 0

                    gesture_text = ""

                    final_gesture = "dont"

                    with yolo_lock:
                        gesture_votes = {}
                        yolo_result = ("Not sure...", 0.0, None)
                        # Clear queue from any leftover frames
                        try:
                            while not yolo_queue.empty():
                                yolo_queue.get_nowait()
                                yolo_queue.task_done()
                        except queue.Empty:
                            pass

                    gesture_frame_count = 0

                    gesture_start_time = time.time()

                    joystick_active = False

                    send_tcp("gesture")

                    print("[MODE] -> GESTURE")

        elif MODE == "CONTROL":
            HOLD_FRAMES = 0

        # ============================================
        # CONTROL MODE
        # ============================================
        if MODE == "CONTROL":

            # Draw fixed joystick guide (background HUD)
            cv2.circle(frame, fixed_joystick_center, JOYSTICK_RADIUS, (100, 100, 100), 1)
            cv2.circle(frame, fixed_joystick_center, DEADZONE, (80, 80, 80), 1)
            cv2.circle(frame, fixed_joystick_center, 5, (0, 255, 255), -1)

            left_hand_detected = False

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                palm = hand_landmarks.landmark[0]

                cx = int(palm.x * w)
                cy = int(palm.y * h)

                palm_center = (cx, cy)

                wrist = hand_landmarks.landmark[0]
                middle = hand_landmarks.landmark[9]

                wx = int(wrist.x * w)
                wy = int(wrist.y * h)

                mx = int(middle.x * w)
                my = int(middle.y * h)

                dynamic_radius = int(
                    distance((wx, wy), (mx, my)) * 1.5
                )

                cv2.circle(
                    frame,
                    palm_center,
                    dynamic_radius,
                    (0, 255, 255),
                    2
                )

                states = finger_states(
                    hand_landmarks,
                    palm_center,
                    dynamic_radius,
                    w,
                    h
                )

                # ============================================
                # LEFT SIDE = MOVE
                # ============================================
                if cx < w // 2:
                    left_hand_detected = True
                    closed = states.count(0)

                    # Apply Exponential Moving Average (EMA) to smooth hand coordinates
                    if smoothed_cx is None:
                        smoothed_cx = cx
                        smoothed_cy = cy
                    else:
                        smoothed_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * smoothed_cx
                        smoothed_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * smoothed_cy

                    scx = int(smoothed_cx)
                    scy = int(smoothed_cy)

                    if closed >= 4:
                        if not joystick_active or dynamic_joystick_center is None:
                            dynamic_joystick_center = (scx, scy)
                        
                        joystick_active = True

                        # Draw active joystick feedback
                        cv2.circle(frame, dynamic_joystick_center, JOYSTICK_RADIUS, (0, 255, 255), 2)
                        cv2.line(frame, dynamic_joystick_center, (scx, scy), (0, 255, 0), 2)
                        cv2.circle(frame, (scx, scy), 8, (0, 255, 0), -1)

                        dx = scx - dynamic_joystick_center[0]
                        dy = scy - dynamic_joystick_center[1]

                        if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
                            left_state = "Idle"
                        else:
                            if abs(dx) > abs(dy):
                                if dx < -DEADZONE:
                                    left_state = "MoveLeft"
                                elif dx > DEADZONE:
                                    left_state = "MoveRight"
                                else:
                                    left_state = "Idle"
                            else:
                                if dy < -DEADZONE:
                                    left_state = "Jump"
                                elif dy > DEADZONE:
                                    left_state = "Crouch"
                                else:
                                    left_state = "Idle"

                    else:
                        joystick_active = False
                        left_state = "Idle"
                        dynamic_joystick_center = None
                        # Reset smoothing when hand is open (not in control/closed state)
                        smoothed_cx = None
                        smoothed_cy = None

                # ============================================
                # RIGHT SIDE = AIM
                # ============================================
                else:
                    thumb_bent = (states[0] == 0)
                    index_bent = (states[1] == 0)
                    middle_bent = (states[2] == 0)
                    ring_bent = (states[3] == 0)
                    pinky_bent = (states[4] == 0)

                    if thumb_bent:
                        if index_bent:
                            right_state = "SHOOT"
                            cv2.putText(frame, "SHOOT", (palm_center[0], palm_center[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        elif middle_bent and ring_bent and pinky_bent:
                            right_state = "AIM"
                            cv2.putText(frame, "AIM", (palm_center[0], palm_center[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        else:
                            right_state = "SWITCH TARGET"
                    else:
                        right_state = "NONE"
                        
            if not left_hand_detected:
                smoothed_cx = None
                smoothed_cy = None

            send_udp(left_state, right_state)

        # ============================================
        # GESTURE MODE
        # ============================================
        elif MODE == "GESTURE":

            elapsed = time.time() - gesture_start_time
            remaining = GESTURE_DURATION - elapsed

            gesture_frame_count += 1
            yolo_input = None

            # Group close hands
            hand_groups = group_hands(results.multi_hand_landmarks, w, h)

            # Process hand groups
            for hand_group in hand_groups:
                # Combine coordinates of all hands in the group
                x_coords = []
                y_coords = []
                for hand_landmarks in hand_group:
                    x_coords.extend([lm.x * w for lm in hand_landmarks.landmark])
                    y_coords.extend([lm.y * h for lm in hand_landmarks.landmark])
                    
                    # Draw landmarks on visual feedback
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                hand_w_raw = max(x_coords) - min(x_coords)
                hand_h_raw = max(y_coords) - min(y_coords)
                hand_size = max(hand_w_raw, hand_h_raw)
                
                # Dynamic square crop calculation (20% padding)
                half_crop_size = hand_size * 0.70
                x_center = (min(x_coords) + max(x_coords)) / 2.0
                y_center = (min(y_coords) + max(y_coords)) / 2.0
                
                x_min = int(x_center - half_crop_size)
                x_max = int(x_center + half_crop_size)
                y_min = int(y_center - half_crop_size)
                y_max = int(y_center + half_crop_size)

                # Clip to frame boundaries
                x_min, x_max = max(x_min, 0), min(x_max, w)
                y_min, y_max = max(y_min, 0), min(y_max, h)

                if x_max > x_min and y_max > y_min:
                    # Crop directly from frame without selfie segmentation
                    hand_roi = frame[y_min:y_max, x_min:x_max]

                    if hand_roi.size > 0:
                        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

                        hand_canvas = np.zeros_like(hand_roi)
                        hand_canvas[:] = (255, 255, 255)
                        mask_3ch = cv2.merge([mask, mask, mask])
                        hand_result = np.where(mask_3ch == 255, hand_canvas, 0)
                        
                        crop_img = cv2.resize(hand_result, (640, 640), interpolation=cv2.INTER_NEAREST)
                        crop_w = x_max - x_min
                        crop_h = y_max - y_min
                        
                        yolo_input = (crop_img.copy(), x_min, y_min, crop_w, crop_h)

                # Draw bounding box on visual feedback per group
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Enqueue to YOLO thread asynchronously (every 3 frames, if space available)
            if yolo_input is not None and gesture_frame_count % 3 == 0:
                try:
                    yolo_queue.put_nowait(yolo_input)
                except queue.Full:
                    pass

            # Read latest prediction from YOLO thread
            with yolo_lock:
                pred_gesture, conf, best_box = yolo_result

            if conf > 0:
                gesture_text = f"{pred_gesture} ({conf:.0%})"
                # Draw YOLO detection on visual feedback frame
                if best_box is not None:
                    x1s, y1s, x2s, y2s = best_box
                    cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), (0, 255, 255), 2)
                    cv2.putText(frame, gesture_text, (x1s, y1s - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                gesture_text = "Not sure..."

            # ============================================
            # COUNTDOWN BAR
            # ============================================
            bar_width = int(
                (remaining / GESTURE_DURATION) * 300
            )

            bar_width = max(0, bar_width)

            cv2.rectangle(
                frame,
                (10, h - 40),
                (310, h - 15),
                (50, 50, 50),
                -1
            )

            cv2.rectangle(
                frame,
                (10, h - 40),
                (10 + bar_width, h - 15),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Back to CONTROL in {max(0, remaining):.1f}s",
                (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            # ============================================
            # FINISH GESTURE
            # ============================================
            if elapsed >= GESTURE_DURATION:

                with yolo_lock:
                    if len(gesture_votes) > 0:

                        final_gesture = max(
                            gesture_votes,
                            key=gesture_votes.get
                        )

                        send_tcp(
                            "",
                            gesture=final_gesture
                        )

                        print(
                            f"[GESTURE] FINAL: {final_gesture}"
                        )

                    else:
                        send_tcp("dont")

                send_tcp("control")

                MODE = "CONTROL"

                HOLD_FRAMES = 0

                gesture_text = ""

                final_gesture = "dont"

                joystick_active = False

                print("[MODE] -> CONTROL")

        # ============================================
        # PD MODE (Finger-Curl Tracking -> Unity)
        # ============================================
        elif MODE == "PD":
            pd_mode_instance.process(frame, results)

    # ============================================
    # NO HAND
    # ============================================
    else:

        HOLD_FRAMES = 0

        joystick_active = False
        dynamic_joystick_center = None

        smoothed_cx = None
        smoothed_cy = None

        send_udp("Idle", "NONE")

    # ============================================
    # UI
    # ============================================
    cv2.rectangle(
        frame,
        (0, 0),
        (640, 60),
        (0, 0, 0),
        -1
    )

    if MODE == "CONTROL":

        cv2.putText(
            frame,
            f"[CONTROL] L:{left_state} R:{right_state}",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2
        )

    elif MODE == "PD":

        cv2.putText(
            frame,
            "[PD] Finger-Curl -> Unity UDP :5052",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (200, 100, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            gesture_text if gesture_text else "Make a gesture...",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 100),
            2
        )

    mode_color = (
        (0, 200, 255) if MODE == "CONTROL"
        else (200, 100, 255) if MODE == "PD"
        else (0, 255, 100)
    )

    cv2.rectangle(
        frame,
        (w - 200, 0),
        (w, 40),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"MODE: {MODE}",
        (w - 195, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        mode_color,
        2
    )

    tcp_status = (
        "TCP: Connected"
        if client_conn
        else "TCP: Waiting..."
    )

    tcp_color = (
        (0, 255, 0)
        if client_conn
        else (0, 0, 255)
    )

    cv2.putText(
        frame,
        tcp_status,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        tcp_color,
        1
    )

    cv2.imshow(
        "Hand Controller + Gesture",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r") and MODE == "PD":
        pd_mode_instance.reset_calibration() # Reset finger smoothing values

# ============================================
# CLEANUP
# ============================================
running = False

# Stop YOLO thread
try:
    yolo_queue.put(None)
    yolo_thread.join(timeout=1.0)
except:
    pass

try:
    if client_conn:
        client_conn.close()
except:
    pass

udp_sock.close()

pd_mode_instance.close()

cap.release()

cv2.destroyAllWindows()