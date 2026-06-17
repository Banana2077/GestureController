import cv2
import mediapipe as mp
import numpy as np
import socket
import json

# ==================== CONFIG ====================
HOST          = "127.0.0.1"
PORT          = 5052
SMOOTHING     = 0.5
CAMERA_ID     = 0

READY_FRAMES  = 45   # ~1.5 วิ ช่วงเตรียมตัว
CALIB_FRAMES  = 60   # ~2 วิ ต่อท่า
# ================================================

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Mapping ปัจจุบัน:
#   thumb  -> ไม่ใช้
#   index  -> ยกแขน (ขึ้น/ลง)
#   middle -> หมุนแขน (หน้า/หลัง)
#   ring   -> ยกขา (ด้านข้าง)
#   pinky  -> ขา (หน้า/หลัง)
LIMB_LABELS = {
    "thumb":  "ไม่ใช้",
    "index":  "ยกแขน (ขึ้น/ลง)",
    "middle": "หมุนแขน (หน้า/หลัง)",
    "ring":   "ยกขา (ด้านข้าง)",
    "pinky":  "ขา (หน้า/หลัง)",
}
DEBUG_COLORS = {
    "thumb": (150,150,150), "index": (50,200,100),
    "middle": (50,150,255), "ring": (50,200,200), "pinky": (255,150,100),
}
FINGER_JOINTS = {
    "thumb":  [2,  3,  4,  4],
    "index":  [5,  6,  7,  8],
    "middle": [9,  10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# ==================== ANGLE CALCULATION ====================

def calc_angle(a, b, c) -> float:
    """คำนวณมุมที่จุด b (องศา) จากสามจุด a->b->c"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

def get_finger_curl(landmarks, name: str) -> float:
    """
    คืนค่าการงอของนิ้ว
    0.0 = เหยียดตรง (มุม ~170°)
    1.0 = งอสุด (มุม ~30°)
    """
    joints = FINGER_JOINTS[name]

    def pt(i):
        return np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z])

    if name == "thumb":
        angle = calc_angle(pt(1), pt(2), pt(4))
        return float(1.0 - np.clip((angle - 30) / 140.0, 0.0, 1.0))
    else:
        mcp, pip, dip, tip = joints
        angle_pip = calc_angle(pt(mcp), pt(pip), pt(dip))
        angle_dip = calc_angle(pt(pip), pt(dip), pt(tip))
        avg_angle = (angle_pip + angle_dip) / 2.0
        return float(1.0 - np.clip((avg_angle - 30) / 140.0, 0.0, 1.0))

def get_raw_values(landmarks) -> dict:
    return {name: get_finger_curl(landmarks, name) for name in FINGER_NAMES}

def is_valid(hand_landmarks) -> bool:
    for i in [0, 4, 8, 12, 16, 20]:
        lm = hand_landmarks.landmark[i]
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return False
    return True

# ==================== CALIBRATION (แยกซ้าย/ขวา) ====================

calib = {
    "Left":  {"min": {n: 1.0 for n in FINGER_NAMES}, "max": {n: 0.0 for n in FINGER_NAMES}},
    "Right": {"min": {n: 1.0 for n in FINGER_NAMES}, "max": {n: 0.0 for n in FINGER_NAMES}},
}
is_calibrated = {"Left": False, "Right": False}

def record_calibration(landmarks, phase: str, hand: str):
    raw = get_raw_values(landmarks)
    for name in FINGER_NAMES:
        if phase == "open":
            calib[hand]["min"][name] = min(calib[hand]["min"][name], raw[name])
        elif phase == "close":
            calib[hand]["max"][name] = max(calib[hand]["max"][name], raw[name])

def normalize(raw: dict, hand: str) -> dict:
    result = {}
    for name in FINGER_NAMES:
        lo = calib[hand]["min"][name]
        hi = calib[hand]["max"][name]
        if hi - lo < 0.01:
            result[name] = 0.0
        else:
            result[name] = float(np.clip((raw[name] - lo) / (hi - lo), 0.0, 1.0))
    return result

# ==================== SMOOTHING (แยกซ้าย/ขวา) ====================

prev_values = {
    "Left":  {name: 0.0 for name in FINGER_NAMES},
    "Right": {name: 0.0 for name in FINGER_NAMES},
}

def smooth(values: dict, hand: str) -> dict:
    prev_values[hand] = {
        name: round(prev_values[hand][name] * SMOOTHING + values[name] * (1 - SMOOTHING), 3)
        for name in FINGER_NAMES
    }
    return prev_values[hand]

# ==================== UDP SENDER ====================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send(left_data, right_data):
    """
    ส่งทั้งสองมือใน packet เดียว
    มือไหนไม่เจอ ส่งเป็น None
    JSON: { "left": {thumb, index, middle, ring, pinky} | null,
            "right": {...} | null }
    """
    payload = json.dumps({"left": left_data, "right": right_data}).encode("utf-8")
    sock.sendto(payload, (HOST, PORT))

# ==================== DEBUG DRAW ====================

def draw_hand_debug(frame, finger_data: dict, hand: str, offset_x: int):
    color_title = (100, 220, 100) if hand == "Left" else (100, 180, 255)
    cv2.putText(frame, f"มือ{('ซ้าย' if hand == 'Left' else 'ขวา')}",
                (offset_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_title, 2)

    for i, name in enumerate(FINGER_NAMES):
        value = finger_data[name]
        color = DEBUG_COLORS[name]
        y     = 60 + i * 28

        label = LIMB_LABELS[name]
        cv2.putText(frame, f"{name} {label}: {value:.2f}",
                    (offset_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        bar_x, bar_w = offset_x, 150
        bar_y = y + 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50,50,50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * value), bar_y + 8), color, -1)
        cv2.line(frame, (bar_x + bar_w, bar_y - 2), (bar_x + bar_w, bar_y + 10), (255,255,255), 1)

def draw_overlay_message(frame, title, msg1, msg2, color, countdown=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    h, w = frame.shape[:2]
    cv2.putText(frame, title, (w//2-160, h//2-80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, msg1, (w//2-180, h//2-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    if msg2:
        cv2.putText(frame, msg2, (w//2-180, h//2+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
    if countdown is not None:
        cv2.putText(frame, f"{countdown}", (w//2-20, h//2+70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

def draw_status(frame, detected_hands: set):
    h = frame.shape[0]
    for i, hand in enumerate(["Left", "Right"]):
        hand_th  = "มือซ้าย" if hand == "Left" else "มือขวา"
        detected = hand in detected_hands
        calib_ok = is_calibrated[hand]

        if detected and calib_ok:
            color, label = (50,200,100), f"{hand_th} ✓"
        elif detected and not calib_ok:
            color, label = (0,165,255), f"{hand_th} (ยังไม่ calibrate)"
        else:
            color, label = (80,80,80), f"{hand_th} ไม่พบ"

        cv2.putText(frame, label, (10 + i * 300, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# ==================== MAIN LOOP ====================

def main():
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(CAMERA_ID)

    # --- Auto-calibration state machine ---
    # "ready" -> "open" -> "close" -> None (เสร็จ)
    calib_phase = "ready"
    calib_timer = 0

    print(f"[Ready] ส่งข้อมูลไปที่ {HOST}:{PORT}")
    print("[Auto-Calibration] กำลังเริ่ม... กด R เพื่อ calibrate ใหม่, Q เพื่อหยุด")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # จัดกลุ่ม landmark ตามมือซ้าย/ขวา
        hand_map = {}
        if results.multi_hand_landmarks:
            for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                           results.multi_handedness):
                label = handedness.classification[0].label
                if is_valid(hand_lm):
                    hand_map[label] = hand_lm
                    mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        detected_hands = set(hand_map.keys())

        # ==================== AUTO CALIBRATION ====================
        if calib_phase == "ready":
            calib_timer += 1
            remaining = max(0, READY_FRAMES - calib_timer) // 15 + 1
            draw_overlay_message(
                frame, "เตรียมตัว Calibrate",
                "ยกมือทั้งสองข้างให้กล้องเห็น",
                "เดี๋ยวจะให้กางมือ และกำมือ",
                (0,165,255), remaining
            )
            if calib_timer >= READY_FRAMES:
                for h in ["Left", "Right"]:
                    calib[h]["min"] = {n: 1.0 for n in FINGER_NAMES}
                    calib[h]["max"] = {n: 0.0 for n in FINGER_NAMES}
                    is_calibrated[h] = False
                calib_phase, calib_timer = "open", 0

        elif calib_phase == "open":
            for hand, hand_lm in hand_map.items():
                record_calibration(hand_lm.landmark, "open", hand)

            remaining = max(0, CALIB_FRAMES - calib_timer)
            draw_overlay_message(
                frame, "ขั้นที่ 1/2",
                "กางมือออกให้สุดทั้งสองข้าง",
                "เหยียดทุกนิ้วให้ตรง",
                (50,200,100), remaining
            )
            calib_timer += 1
            if calib_timer >= CALIB_FRAMES:
                calib_phase, calib_timer = "close", 0

        elif calib_phase == "close":
            for hand, hand_lm in hand_map.items():
                record_calibration(hand_lm.landmark, "close", hand)

            remaining = max(0, CALIB_FRAMES - calib_timer)
            draw_overlay_message(
                frame, "ขั้นที่ 2/2",
                "กำมือให้แน่นทั้งสองข้าง",
                "งอทุกนิ้วให้สุด",
                (50,150,255), remaining
            )
            calib_timer += 1
            if calib_timer >= CALIB_FRAMES:
                for h in ["Left", "Right"]:
                    if calib[h]["max"][FINGER_NAMES[0]] > calib[h]["min"][FINGER_NAMES[0]]:
                        is_calibrated[h] = True
                calib_phase = None
                print(f"[Calibration Done] {is_calibrated}")

        # ==================== NORMAL TRACKING ====================
        else:
            left_data  = None
            right_data = None

            for hand, hand_lm in hand_map.items():
                raw         = get_raw_values(hand_lm.landmark)
                normalized  = normalize(raw, hand) if is_calibrated[hand] else raw
                finger_data = smooth(normalized, hand)

                offset_x = 10 if hand == "Left" else frame.shape[1] // 2 + 10
                draw_hand_debug(frame, finger_data, hand, offset_x)

                if hand == "Left":
                    left_data  = finger_data
                else:
                    right_data = finger_data

            send(left_data, right_data)
            draw_status(frame, detected_hands)

        cv2.imshow("Hand Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            calib_phase, calib_timer = "ready", 0
            print("[Auto-Calibration] เริ่มใหม่...")

    cap.release()
    hands.close()
    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()