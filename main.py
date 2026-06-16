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
# LOAD MODEL
# ============================================
with open("model.pkl", "rb") as f:
    model, le, scaler = pickle.load(f)

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
    "bird": "bird",
    "frog": "frog",
}

CONFIDENCE_THRESHOLD = 0.5

# ============================================
# MODE
# ============================================
MODE = "CONTROL"

# ============================================
# STATE
# ============================================
HOLD_FRAMES = 0
HOLD_REQUIRED = 5

gesture_text = ""
final_gesture = "dont"

GESTURE_DURATION = 0.75
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

last_tcp_message = ""

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

    left_state = "Idle"
    right_state = "NONE"

    # ============================================
    # DETECT HAND
    # ============================================
    if results.multi_hand_landmarks:

        all_open = all(
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

                MODE = "GESTURE"

                HOLD_FRAMES = 0

                gesture_text = ""

                final_gesture = "dont"

                gesture_votes = {}

                gesture_start_time = time.time()

                joystick_active = False

                send_tcp("gesture")

                print("[MODE] → GESTURE")

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
                            right_state = "SWITCH_TARGET"
                            cv2.putText(frame, "SWITCH TARGET", (palm_center[0], palm_center[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        else:
                            right_state = "AIM"
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

            row = []

            for hand_landmarks in results.multi_hand_landmarks:

                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min = min(x_coords)
                x_max = max(x_coords)

                y_min = min(y_coords)
                y_max = max(y_coords)

                box_w = x_max - x_min
                box_h = y_max - y_min

                pad = 0.02

                x1 = int((x_min - pad) * w)
                y1 = int((y_min - pad) * h)

                x2 = int((x_max + pad) * w)
                y2 = int((y_max + pad) * h)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for lm in hand_landmarks.landmark:

                    x_norm = (
                        (lm.x - x_min) / box_w
                        if box_w > 0 else 0.0
                    )

                    y_norm = (
                        (lm.y - y_min) / box_h
                        if box_h > 0 else 0.0
                    )

                    row.extend([
                        x_norm,
                        y_norm,
                        lm.z
                    ])

            if len(results.multi_hand_landmarks) == 1:
                row.extend([0.0] * 63)

            if len(row) == 126:

                row_scaled = scaler.transform([row])

                proba = model.predict_proba(row_scaled)[0]

                confidence_val = proba.max()

                pred_index = proba.argmax()

                pred_name = le.inverse_transform([pred_index])[0]

                if confidence_val >= CONFIDENCE_THRESHOLD:

                    gesture_text = (
                        f"{pred_name} ({confidence_val:.0%})"
                    )

                    gesture_votes[pred_name] = (
                        gesture_votes.get(pred_name, 0) + 1
                    )

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

                print("[MODE] → CONTROL")

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
        (0, 200, 255)
        if MODE == "CONTROL"
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

# ============================================
# CLEANUP
# ============================================
running = False

try:
    if client_conn:
        client_conn.close()
except:
    pass

udp_sock.close()

cap.release()

cv2.destroyAllWindows()