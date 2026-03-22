import cv2
import mediapipe as mp
import pickle
import socket
import threading

# ───── โหลดโมเดล ─────
with open("model.pkl", "rb") as f:
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

GESTURE_LABELS = {
    "rabbit": "rabbit",
    "bird":   "bird",
    "frog":   "frog",
}

CONFIDENCE_THRESHOLD = 0.5

# ───── State ─────
STATE         = "WAIT"
HOLD_FRAMES   = 0
HOLD_REQUIRED = 50
gesture_text  = ""

# ───── Socket Server ─────
HOST = "localhost"
PORT = 5005

client_conn = None
client_lock = threading.Lock()

def socket_server():
    global client_conn
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[Socket] Waiting for Unity to connect at {HOST}:{PORT} ...")
    while True:
        conn, addr = server.accept()
        with client_lock:
            client_conn = conn
        print(f"[Socket]  Unity connected: {addr}")

def send_to_unity(status, gesture=""):
    """
    format ที่ส่ง:
      1:dont        → ยังไม่พบมือ / รอแบมือ
      2:rabbit      → จำแนกได้แล้ว
    """
    global client_conn
    if gesture:
        message = f"2:{gesture}"
    else:
        message = f"1:{status}"

    with client_lock:
        if client_conn:
            try:
                client_conn.sendall((message + "\n").encode("utf-8"))
                print(f"[Socket] Sending to Unity: {message}")
            except:
                client_conn = None
                print("[Socket] Unity disconnected")

server_thread = threading.Thread(target=socket_server, daemon=True)
server_thread.start()

def is_open_hand(hand_landmarks):
    tips    = [8, 12, 16, 20]
    middles = [6, 10, 14, 18]
    fingers_open = 0
    for tip, mid in zip(tips, middles):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mid].y:
            fingers_open += 1
    return fingers_open >= 4

last_sent = ""  # ป้องกันส่งซ้ำ

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    current_status = "dont"  # default = dont

    if results.multi_hand_landmarks:
        row = []

        for hand_landmarks in results.multi_hand_landmarks:

            # ─── Bounding Box ───
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            box_w = x_max - x_min
            box_h = y_max - y_min

            pad = 0.02
            x1 = int((x_min - pad) * w)
            y1 = int((y_min - pad) * h)
            x2 = int((x_max + pad) * w)
            y2 = int((y_max + pad) * h)

            box_color = (0, 255, 0) if STATE == "READY" else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx == 0:
                    color = (255, 255, 255)
                elif idx in [1, 2, 3, 4]:
                    color = (255, 100, 100)
                elif idx in [5, 6, 7, 8]:
                    color = (100, 255, 100)
                elif idx in [9, 10, 11, 12]:
                    color = (100, 100, 255)
                elif idx in [13, 14, 15, 16]:
                    color = (255, 255, 100)
                else:
                    color = (255, 100, 255)
                cv2.circle(frame, (cx, cy), 6, color, -1)
                cv2.putText(frame, str(idx), (cx + 8, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            for lm in hand_landmarks.landmark:
                x_norm = (lm.x - x_min) / box_w if box_w > 0 else 0.0
                y_norm = (lm.y - y_min) / box_h if box_h > 0 else 0.0
                row.extend([x_norm, y_norm, lm.z])

        if len(results.multi_hand_landmarks) == 1:
            row.extend([0.0] * 63)

        # ───── State Machine ─────
        all_open = all(
            is_open_hand(hl) for hl in results.multi_hand_landmarks
        )

        if STATE == "WAIT":
            current_status = "dont"     # เจอมือแต่ยังรอแบมือ
            if all_open:
                HOLD_FRAMES += 1
                progress = int((HOLD_FRAMES / HOLD_REQUIRED) * 200)
                cv2.rectangle(frame, (10, h-40), (210, h-15), (50,50,50), -1)
                cv2.rectangle(frame, (10, h-40), (10+progress, h-15), (0,255,255), -1)
                cv2.putText(frame, "Hold your hand...",
                            (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                if HOLD_FRAMES >= HOLD_REQUIRED:
                    STATE = "READY"
                    HOLD_FRAMES = 0
            else:
                HOLD_FRAMES = 0

        elif STATE == "READY":
            if len(row) == 126:
                row_scaled = scaler.transform([row])
                proba = model.predict_proba(row_scaled)[0]
                confidence_val = proba.max()
                pred_index = proba.argmax()
                pred_name = le.inverse_transform([pred_index])[0]

                if confidence_val >= CONFIDENCE_THRESHOLD:
                    label = GESTURE_LABELS.get(pred_name, pred_name)
                    gesture_text = f"{label}  ({confidence_val:.0%})"
                    current_status = pred_name  # เช่น "rabbit"
                else:
                    gesture_text  = "Not sure..."
                    current_status = "dont"

    else:
        # ไม่พบมือเลย
        current_status = "dont"
        STATE       = "WAIT"
        HOLD_FRAMES = 0
        last_sent   = ""

    # ───── ส่งไป Unity เฉพาะเมื่อสถานะเปลี่ยน ─────
    if current_status != last_sent:
        if current_status == "dont":
            send_to_unity("dont")
        else:
            send_to_unity("", gesture=current_status)
        last_sent = current_status

    # ───── แสดงผล ─────
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)

    if STATE == "WAIT":
        cv2.putText(frame, "Hold your hand to start",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    else:
        cv2.putText(frame,
                    gesture_text if gesture_text else "Make a gesture...",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)

    socket_status = "Unity connected" if client_conn else "Waiting for Unity..."
    socket_color  = (0, 255, 0) if client_conn else (0, 0, 255)
    cv2.putText(frame, socket_status,
                (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, socket_color, 2)
    cv2.putText(frame, "SPACE = Reset    Q = ออก",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    cv2.imshow("Hand Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        STATE        = "WAIT"
        HOLD_FRAMES  = 0
        gesture_text = ""
        last_sent    = ""
        send_to_unity("dont")
        print("Reset!")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()