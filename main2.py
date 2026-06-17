import cv2
import mediapipe as mp
import pickle
import socket
import threading
import math

# ───── โหลดโมเดล ─────
with open(r"C:\Users\comsc\Documents\GitHub\GestureController\model.pkl", "rb") as f:
    model, le, scaler = pickle.load(f)

# ───── ตั้งค่า MediaPipe ─────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ───── Gesture Labels ─────
GESTURE_LABELS = {
    "rabbit": "rabbit",
    "bird":   "bird",
    "frog":   "frog",
}

CONFIDENCE_THRESHOLD = 0.5

# ───── โหมด ─────
MODE = "CONTROL"

# ───── State ─────
HOLD_FRAMES      = 0
HOLD_REQUIRED    = 20
gesture_text     = ""
GESTURE_DURATION = 6.0
gesture_timer    = 0.0
final_gesture    = "dont"

# ───── Joystick ─────
JOYSTICK_RADIUS = 200
DEADZONE        = 40
joystick_center = None
joystick_active = False

# ───── TCP (Gesture → Unity) ─────
TCP_HOST    = "localhost"
TCP_PORT    = 5005
client_conn = None
client_lock = threading.Lock()

def socket_server():
    global client_conn
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(1)
    print(f"[TCP] Waiting for Unity at {TCP_HOST}:{TCP_PORT} ...")
    while True:
        conn, addr = server.accept()
        with client_lock:
            client_conn = conn
        print(f"[TCP] Unity connected: {addr}")

server_thread = threading.Thread(target=socket_server, daemon=True)
server_thread.start()

def send_tcp(status, gesture=""):
    global client_conn

    if gesture:
        message = f"2:{gesture}"     # จำแนกท่าได้
    elif status in ["control", "gesture"]:
        message = f"0:{status}"      # ← เปลี่ยนโหมด
    else:
        message = f"1:{status}"      # dont

    with client_lock:
        if client_conn:
            try:
                client_conn.sendall((message + "\n").encode("utf-8"))
                print(f"[TCP] Sent: {message}")
            except:
                client_conn = None

# ───── UDP (Control → Unity) ─────
UDP_HOST = "127.0.0.1"
UDP_PORT = 5006
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_udp(left_state, right_state):
    message = f"L:{left_state}\nR:{right_state}"
    udp_sock.sendto(message.encode(), (UDP_HOST, UDP_PORT))

# ───── Helper ─────
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def is_open_hand(hand_landmarks):
    tips    = [8, 12, 16, 20]
    middles = [6, 10, 14, 18]
    count   = 0
    for tip, mid in zip(tips, middles):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mid].y:
            count += 1
    return count >= 4

def finger_states(hand_landmarks, palm_center, radius, w, h):
    states     = []
    finger_ids = [4, 8, 12, 16, 20]
    for i in finger_ids:
        tip = hand_landmarks.landmark[i]
        x   = int(tip.x * w)
        y   = int(tip.y * h)
        d   = distance((x, y), palm_center)
        states.append(0 if d < radius else 1)
    return states

last_sent = ""

# ─────────────────────────────────
#              MAIN LOOP
# ─────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    left_state     = "IDLE"
    right_state    = "NONE"
    current_status = "dont"

    if results.multi_hand_landmarks:

        all_open = all(is_open_hand(hl) for hl in results.multi_hand_landmarks)

        # ───── เช็คแบมือ → สลับ CONTROL → GESTURE ─────
        if all_open and MODE == "CONTROL":
            HOLD_FRAMES += 1
            progress = int((HOLD_FRAMES / HOLD_REQUIRED) * 200)
            cv2.rectangle(frame, (10, h-40), (210, h-15), (50,50,50), -1)
            cv2.rectangle(frame, (10, h-40), (10+progress, h-15), (0,255,255), -1)
            cv2.putText(frame, "Hold to switch GESTURE mode...",
                        (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
            if HOLD_FRAMES >= HOLD_REQUIRED:
                MODE            = "GESTURE"
                HOLD_FRAMES     = 0
                gesture_text    = ""
                last_sent       = ""
                gesture_timer   = 0.0
                joystick_active = False
                send_tcp("gesture")
                print("[MODE] → GESTURE MODE")
        elif not all_open or MODE == "GESTURE":
            if MODE == "CONTROL":
                HOLD_FRAMES = 0

        # ══════════════════════════════
        #  CONTROL MODE
        # ══════════════════════════════
        if MODE == "CONTROL":
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label = handedness.classification[0].label

                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                palm        = hand_landmarks.landmark[0]
                cx          = int(palm.x * w)
                cy          = int(palm.y * h)
                palm_center = (cx, cy)

                wrist  = hand_landmarks.landmark[0]
                middle = hand_landmarks.landmark[9]
                wx, wy = int(wrist.x * w),  int(wrist.y * h)
                mx, my = int(middle.x * w), int(middle.y * h)
                dynamic_radius = int(distance((wx, wy), (mx, my)) * 1.5)

                cv2.circle(frame, palm_center, dynamic_radius, (0,255,255), 2)

                states = finger_states(hand_landmarks, palm_center, dynamic_radius, w, h)

                # ─── มือซ้าย → เดิน ───
                if label == "Left":
                    closed = states.count(0)
                    if closed >= 4:
                        if not joystick_active:
                            joystick_center = palm_center
                            joystick_active = True

                        cv2.circle(frame, joystick_center, JOYSTICK_RADIUS, (255,255,0), 2)
                        dx = cx - joystick_center[0]
                        dy = cy - joystick_center[1]

                        if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
                            left_state = "MoveForward"
                        elif dx < -DEADZONE:
                            left_state = "MoveLeft"
                        elif dx > DEADZONE:
                            left_state = "MoveRight"
                        elif dy < -DEADZONE:
                            left_state = "Jump"
                        elif dy > DEADZONE:
                            left_state = "Crouch"
                    else:
                        joystick_active = False
                        left_state      = "IDLE"

                # ─── มือขวา → เล็ง ───
                if label == "Right":
                    index_tip  = hand_landmarks.landmark[8]
                    index_base = hand_landmarks.landmark[5]
                    tx = int(index_tip.x * w)
                    ty = int(index_tip.y * h)
                    bx = int(index_base.x * w)
                    by = int(index_base.y * h)
                    dx = tx - bx
                    dy = ty - by

                    cv2.arrowedLine(frame, (bx, by), (tx, ty), (0,255,0), 3)

                    if abs(dx) > abs(dy):
                        if dx > 20:    right_state = "AIM_RIGHT"
                        elif dx < -20: right_state = "AIM_LEFT"
                    else:
                        if dy > 20:    right_state = "AIM_DOWN"
                        elif dy < -20: right_state = "AIM_UP"

            send_udp(left_state, right_state)

        # ══════════════════════════════
        #  GESTURE MODE
        # ══════════════════════════════
        elif MODE == "GESTURE":

            gesture_timer += 1 / 30
            remaining      = GESTURE_DURATION - gesture_timer

            row = []

            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                box_w = x_max - x_min
                box_h = y_max - y_min

                pad = 0.02
                x1  = int((x_min - pad) * w)
                y1  = int((y_min - pad) * h)
                x2  = int((x_max + pad) * w)
                y2  = int((y_max + pad) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for lm in hand_landmarks.landmark:
                    x_norm = (lm.x - x_min) / box_w if box_w > 0 else 0.0
                    y_norm = (lm.y - y_min) / box_h if box_h > 0 else 0.0
                    row.extend([x_norm, y_norm, lm.z])

            if len(results.multi_hand_landmarks) == 1:
                row.extend([0.0] * 63)

            if len(row) == 126:
                row_scaled     = scaler.transform([row])
                proba          = model.predict_proba(row_scaled)[0]
                confidence_val = proba.max()
                pred_index     = proba.argmax()
                pred_name      = le.inverse_transform([pred_index])[0]

                if confidence_val >= CONFIDENCE_THRESHOLD:
                    label          = GESTURE_LABELS.get(pred_name, pred_name)
                    gesture_text   = f"{label}  ({confidence_val:.0%})"
                    final_gesture  = pred_name   # ← แค่เก็บไว้ ยังไม่ส่ง
                else:
                    gesture_text  = "Not sure..."

            # ─── countdown bar ───
            bar_width = int((remaining / GESTURE_DURATION) * 300)
            bar_width = max(0, bar_width)
            bar_color = (0, 255, 0) if remaining > 4 else (0, 100, 255)
            cv2.rectangle(frame, (10, h-40), (310, h-15), (50,50,50), -1)
            cv2.rectangle(frame, (10, h-40), (10+bar_width, h-15), bar_color, -1)
            cv2.putText(frame, f"Back to CONTROL in {max(0, remaining):.1f}s",
                        (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 2)

            # ─── ครบ 10 วิ → ส่งครั้งเดียว แล้วกลับ CONTROL ───
            if gesture_timer >= GESTURE_DURATION:

                # ส่งแค่ครั้งเดียวตรงนี้
                if final_gesture != "dont":
                    send_tcp("", gesture=final_gesture)
                    print(f"[TCP] ส่ง gesture สุดท้าย: {final_gesture}")
                else:
                    send_tcp("dont")

                send_tcp("control")   # บอก Unity ว่ากลับ CONTROL แล้ว

                MODE          = "CONTROL"
                gesture_timer = 0.0
                HOLD_FRAMES   = 0
                gesture_text  = ""
                final_gesture = "dont"   # reset
                last_sent     = ""
                print("[MODE] → CONTROL MODE (timeout)")
    else:
        HOLD_FRAMES     = 0
        joystick_active = False
        last_sent       = ""
        if MODE == "GESTURE":
            send_tcp("dont")
        send_udp("IDLE", "NONE")

    # ══════════════════════════════
    #  UI
    # ══════════════════════════════
    cv2.rectangle(frame, (0, 0), (640, 60), (0,0,0), -1)

    if MODE == "CONTROL":
        cv2.putText(frame, f"[CONTROL]  L:{left_state}  R:{right_state}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
    else:
        cv2.putText(frame,
                    gesture_text if gesture_text else "Make a gesture...",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,100), 2)

    mode_color = (0,200,255) if MODE == "CONTROL" else (0,255,100)
    cv2.rectangle(frame, (w-200, 0), (w, 40), (0,0,0), -1)
    cv2.putText(frame, f"MODE: {MODE}",
                (w-195, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    tcp_status = "TCP: Connected" if client_conn else "TCP: Waiting..."
    tcp_color  = (0,255,0) if client_conn else (0,0,255)
    cv2.putText(frame, tcp_status,
                (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tcp_color, 1)
    cv2.putText(frame, "Hold open hand = GESTURE Mode  |  Q = Quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    cv2.imshow("Hand Controller + Gesture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        gesture_text  = ""
        gesture_timer = 0.0
        last_sent     = ""
        send_tcp("dont")
        print("Reset!")

cap.release()
cv2.destroyAllWindows()