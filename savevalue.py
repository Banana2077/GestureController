import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

GESTURE_NAME = "frog"   # ← เปลี่ยนตามท่าที่เก็บแต่ละรอบ
OUTPUT_FILE = "dataset.csv"
COUNT = 0
collecting = False

with open(OUTPUT_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            row = []

            for hand_landmarks in results.multi_hand_landmarks:

                # ─── คำนวณ Bounding Box ───
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                box_w = x_max - x_min
                box_h = y_max - y_min

                # ─── วาดกรอบรอบมือ ───
                pad = 0.02
                x1 = int((x_min - pad) * w)
                y1 = int((y_min - pad) * h)
                x2 = int((x_max + pad) * w)
                y2 = int((y_max + pad) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # ─── วาด Landmark + หมายเลข ───
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
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

                # ─── คำนวณพิกัดภายใน Bounding Box ───
                for lm in hand_landmarks.landmark:
                    x_norm = (lm.x - x_min) / box_w if box_w > 0 else 0.0
                    y_norm = (lm.y - y_min) / box_h if box_h > 0 else 0.0
                    row.extend([x_norm, y_norm, lm.z])

            # ─── Padding ถ้ามีมือแค่ข้างเดียว ───
            if len(results.multi_hand_landmarks) == 1:
                row.extend([0.0] * 63)

            # ─── บันทึกถ้ากำลัง Record ───
            if collecting and len(row) == 126:
                writer.writerow([GESTURE_NAME] + row)
                COUNT += 1

        # ─── UI สถานะ ───
        status_color = (0, 0, 200) if collecting else (100, 100, 100)
        status_text = f"REC  {COUNT} frames" if collecting else "PAUSE"
        cv2.rectangle(frame, (0, 0), (420, 55), (0, 0, 0), -1)
        cv2.putText(frame, f"{status_text}  |  Gesture: {GESTURE_NAME}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, "SPACE = REC/PAUSE    Q = ออก",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            collecting = not collecting
            if not collecting:
                print(f"หยุดเก็บ — รวม {COUNT} frames สำหรับท่า '{GESTURE_NAME}'")
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"เสร็จสิ้น! เก็บได้ {COUNT} frames สำหรับท่า '{GESTURE_NAME}'")