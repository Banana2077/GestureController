"""
pd_mode.py  -  Finger-Curl Tracking Mode (PD Mode) with Auto-Calibration
======================================================================
Tracks raw finger curl values (0.0 to 1.0) using Mediapipe and sends them
directly to Unity via UDP on port 5052. Includes an automatic calibration
mechanism that dynamically learns each user's finger ranges during runtime.

Protocol
--------
Sends a JSON packet per frame:
    {
        "left":  {"thumb": 0.0, "index": 0.52, "middle": 0.0, "ring": 0.0, "pinky": 0.0} | null,
        "right": {...} | null
    }
"""

import json
import socket
import numpy as np

# ==================== CONFIG ====================
PD_HOST      = "127.0.0.1"
PD_PORT      = 5052
PD_SMOOTHING = 0.5

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

LIMB_LABELS = {
    "thumb":  "Unused",
    "index":  "Arm (Up/Down)",
    "middle": "Arm Swing (Fwd/Back)",
    "ring":   "Leg Lift (Side)",
    "pinky":  "Leg Swing (Fwd/Back)",
}

DEBUG_COLORS = {
    "thumb":  (150, 150, 150),
    "index":  (50,  200, 100),
    "middle": (50,  150, 255),
    "ring":   (50,  200, 200),
    "pinky":  (255, 150, 100),
}

FINGER_JOINTS = {
    "thumb":  [2,  3,  4,  4],
    "index":  [5,  6,  7,  8],
    "middle": [9,  10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# กำหนดขอบเขตความกว้างองศาเริ่มต้นเพื่อเซฟและป้องกันระบบแกว่งตัวในเฟรมแรกๆ
# (เมื่อผู้ใช้ขยับมือ นิ้วที่เหยียด/งอเกินค่านี้ ระบบจะอัปเดตขยายขอบเขตอัตโนมัติ)
DEFAULT_MIN_ANGLE = 100.0  # ค่าเริ่มต้นสำหรับงอนิ้ว (จะค่อยๆ ปรับลดลงตามความงอจริง)
DEFAULT_MAX_ANGLE = 120.0  # ค่าเริ่มต้นสำหรับเหยียดนิ้ว (จะค่อยๆ ปรับเพิ่มขึ้นตามความเหยียดจริง)


# ==================== MATH HELPERS ====================

def _calc_angle(a, b, c) -> float:
    """Compute angle at point b (degrees) from three points a->b->c"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _get_finger_angle(landmarks, name: str) -> float:
    """
    คืนค่ามุมองศาจริงของนิ้ว (องศา)
    - นิ้วทั่วไป: หามุมเฉลี่ยระหว่างข้อต่อ PIP และ DIP
    - นิ้วโป้ง: หามุมข้อต่อ MCP (จุด 1-2-4)
    """
    def pt(i):
        return np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z])

    if name == "thumb":
        return _calc_angle(pt(1), pt(2), pt(4))
    else:
        mcp, pip, dip, tip = FINGER_JOINTS[name]
        angle_pip = _calc_angle(pt(mcp), pt(pip), pt(dip))
        angle_dip = _calc_angle(pt(pip), pt(dip), pt(tip))
        return float((angle_pip + angle_dip) / 2.0)


def _is_valid(hand_landmarks) -> bool:
    for i in [0, 4, 8, 12, 16, 20]:
        lm = hand_landmarks.landmark[i]
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return False
    return True


# ==================== PD MODE CLASS ====================

class PDMode:
    def __init__(self, host=PD_HOST, port=PD_PORT, smoothing=PD_SMOOTHING):
        self.host      = host
        self.port      = port
        self.smoothing = smoothing

        # UDP socket for sending data to Unity
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Smoothing state (per hand)
        self._prev = {
            "Left":  {n: 0.0 for n in FINGER_NAMES},
            "Right": {n: 0.0 for n in FINGER_NAMES},
        }

        # เก็บสถานะการ Calibrate ขอบเขตองศา (แยกมือซ้าย-ขวา และแยกรายนิ้ว)
        self._calib = {
            "Left":  {n: {"min": DEFAULT_MIN_ANGLE, "max": DEFAULT_MAX_ANGLE} for n in FINGER_NAMES},
            "Right": {n: {"min": DEFAULT_MIN_ANGLE, "max": DEFAULT_MAX_ANGLE} for n in FINGER_NAMES},
        }

        print(f"[PD Mode] Initialized with Auto-Calibration. UDP -> {host}:{port}")

    def process(self, frame, results):
        """
        เรียกใช้ทุกเฟรมในขณะที่ MODE == "PD"
        """
        import cv2

        h_frame, w_frame, _ = frame.shape

        # จัดกลุ่ม Landmark โดยแยกมือซ้าย/มือขวา
        hand_map = {}
        if results and results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                           results.multi_handedness):
                label = handedness.classification[0].label
                if _is_valid(hand_lm):
                    hand_map[label] = hand_lm
                    # วาดโครงกระดูกมือบนเฟรม
                    import mediapipe as mp
                    mp_draw = mp.solutions.drawing_utils
                    mp_draw.draw_landmarks(
                        frame, hand_lm,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )

        detected_hands = set(hand_map.keys())

        left_data  = None
        right_data = None

        for hand in ["Left", "Right"]:
            if hand in hand_map:
                hand_lm = hand_map[hand]
                
                # 1. ดึงมุมจริง ณ เฟรมปัจจุบันของแต่ละนิ้ว
                raw_angles = {name: _get_finger_angle(hand_lm.landmark, name) for name in FINGER_NAMES}
                
                # 2. Auto-Calibrate: อัปเดตช่วงขอบเขตสูงสุด-ต่ำสุดอย่างชาญฉลาด
                for name in FINGER_NAMES:
                    angle = raw_angles[name]
                    # ถว่างอนิ้วได้ลึกกว่าค่าเดิมที่จำไว้ -> อัปเดตขยายขอบเขต min ลงไปอีก
                    if angle < self._calib[hand][name]["min"]:
                        self._calib[hand][name]["min"] = angle
                    # ถ้ายืดนิ้วได้ตรงกว่าค่าเดิมที่จำไว้ -> อัปเดตขยายขอบเขต max ขึ้นไปอีก
                    if angle > self._calib[hand][name]["max"]:
                        self._calib[hand][name]["max"] = angle

                # 3. แปลงค่าองศาให้เป็นค่าสากล (0.0 = นิ้วเหยียดตรง, 1.0 = นิ้วงอสุด)
                normalized = {}
                for name in FINGER_NAMES:
                    angle = raw_angles[name]
                    min_a = self._calib[hand][name]["min"]
                    max_a = self._calib[hand][name]["max"]

                    if max_a == min_a:
                        curl = 0.0
                    else:
                        # ถ้ายิ่งใกล้มุมต่ำ (min_a) แสดงว่างอสุด (1.0) ถ้ายิ่งใกล้มุมสูง (max_a) แสดงว่ายืดสุด (0.0)
                        curl = (max_a - angle) / (max_a - min_a)

                    normalized[name] = float(np.clip(curl, 0.0, 1.0))

                # 4. ใช้ EMA กรองความสมูทในการส่งค่าขยับนิ้ว
                finger_data = self._smooth(normalized, hand)

                offset_x = 10 if hand == "Left" else w_frame // 2 + 10
                self._draw_hand_debug(frame, finger_data, hand, offset_x)

                if hand == "Left":
                    left_data  = finger_data
                else:
                    right_data = finger_data

        self._send(left_data, right_data)
        self._draw_status(frame, detected_hands)
        return "tracking"

    def reset_calibration(self):
        """ล้างประวัติการจำค่าความเรียบเนียนและการตั้งค่าคาลิเบรตนิ้วทั้งหมด"""
        self._prev = {
            "Left":  {n: 0.0 for n in FINGER_NAMES},
            "Right": {n: 0.0 for n in FINGER_NAMES},
        }
        self._calib = {
            "Left":  {n: {"min": DEFAULT_MIN_ANGLE, "max": DEFAULT_MAX_ANGLE} for n in FINGER_NAMES},
            "Right": {n: {"min": DEFAULT_MIN_ANGLE, "max": DEFAULT_MAX_ANGLE} for n in FINGER_NAMES},
        }
        print("[PD Mode] Reset finger smoothing and auto-calibration limits.")

    def close(self):
        """ปิดการเชื่อมต่อ UDP socket"""
        try:
            self._sock.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PRIVATE - SMOOTHING
    # ------------------------------------------------------------------

    def _smooth(self, values: dict, hand: str) -> dict:
        alpha = self.smoothing
        self._prev[hand] = {
            name: round(self._prev[hand][name] * alpha + values[name] * (1 - alpha), 3)
            for name in FINGER_NAMES
        }
        return self._prev[hand]

    # ------------------------------------------------------------------
    # PRIVATE - UDP SEND
    # ------------------------------------------------------------------

    def _send(self, left_data, right_data):
        payload = json.dumps({"left": left_data, "right": right_data}).encode("utf-8")
        try:
            self._sock.sendto(payload, (self.host, self.port))
        except Exception as e:
            print(f"[PD Mode UDP Error] {e}")

    # ------------------------------------------------------------------
    # PRIVATE - DRAW HELPERS
    # ------------------------------------------------------------------

    def _draw_hand_debug(self, frame, finger_data: dict, hand: str, offset_x: int):
        import cv2
        color_title = (100, 220, 100) if hand == "Left" else (100, 180, 255)
        hand_label  = "Left Hand" if hand == "Left" else "Right Hand"
        cv2.putText(frame, hand_label,
                    (offset_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_title, 2)

        for i, name in enumerate(FINGER_NAMES):
            value  = finger_data[name]
            color  = DEBUG_COLORS[name]
            y      = 108 + i * 28
            label  = LIMB_LABELS[name]
            
            # ดึงช่วง Calibrated ปัจจุบันมาแสดงผลบนหน้าจอดีบัก
            min_a  = self._calib[hand][name]["min"]
            max_a  = self._calib[hand][name]["max"]

            # แสดงช่วงองศาจริงที่จับได้ (เช่น 84-152 องศา) ข้างชื่อนิ้ว
            cv2.putText(frame, f"{name} ({min_a:.0f}-{max_a:.0f}deg): {value:.2f}",
                        (offset_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

            bar_x, bar_w = offset_x, 130
            bar_y = y + 5
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w * value), bar_y + 8), color, -1)
            cv2.line(frame, (bar_x + bar_w, bar_y - 2),
                     (bar_x + bar_w, bar_y + 10), (255, 255, 255), 1)

    def _draw_status(self, frame, detected_hands: set):
        import cv2
        frame_h = frame.shape[0]
        for i, hand in enumerate(["Left", "Right"]):
            hand_label = "Left Hand" if hand == "Left" else "Right Hand"
            detected   = hand in detected_hands

            if detected:
                color, label = (50, 200, 100), f"{hand_label} Detected"
            else:
                color, label = (80, 80, 80), f"{hand_label} Not found"

            cv2.putText(frame, label,
                        (10 + i * 300, frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)