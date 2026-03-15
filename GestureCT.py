import cv2
import mediapipe as mp
import socket
import math

# =========================
# UDP SETUP
# =========================

HOST = "127.0.0.1"
PORT = 5055

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# =========================
# MEDIAPIPE SETUP
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

# =========================
# JOYSTICK SETTINGS
# =========================

JOYSTICK_RADIUS = 200
DEADZONE = 40

joystick_center = None
joystick_active = False

# =========================
# STATE VALUES
# =========================

left_state = "IDLE"
right_state = "NONE"

# =========================
# HELPER FUNCTIONS
# =========================

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def send_udp():

    message = f"L:{left_state}\nR:{right_state}"

    sock.sendto(message.encode(), (HOST, PORT))

    print(message)
    print("-----")


def finger_states(hand_landmarks, palm_center, radius):

    states = []

    finger_ids = [4,8,12,16,20]

    for i in finger_ids:

        tip = hand_landmarks.landmark[i]

        x = int(tip.x * width)
        y = int(tip.y * height)

        d = distance((x,y), palm_center)

        if d < radius:
            states.append(0)
        else:
            states.append(1)

    return states


# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame,1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    left_state = "IDLE"
    right_state = "NONE"

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):

            label = handedness.classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # =========================
            # PALM CENTER
            # =========================

            palm = hand_landmarks.landmark[0]

            cx = int(palm.x * width)
            cy = int(palm.y * height)

            palm_center = (cx,cy)

            # =========================
            # DYNAMIC RADIUS
            # =========================

            wrist = hand_landmarks.landmark[0]
            middle = hand_landmarks.landmark[9]

            wx = int(wrist.x * width)
            wy = int(wrist.y * height)

            mx = int(middle.x * width)
            my = int(middle.y * height)

            dynamic_radius = int(distance((wx,wy),(mx,my))*1.5)

            cv2.circle(frame,palm_center,dynamic_radius,(0,255,255),2)

            # =========================
            # FINGER CHECK
            # =========================

            states = finger_states(hand_landmarks,palm_center,dynamic_radius)

            # =========================
            # LEFT HAND → MOVEMENT
            # =========================

            if label == "Left":

                closed = states.count(0)

                if closed >= 4:

                    if not joystick_active:
                        joystick_center = palm_center
                        joystick_active = True

                    cv2.circle(frame,joystick_center,JOYSTICK_RADIUS,(255,255,0),2)

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
                    left_state = "IDLE"

            # =========================
            # RIGHT HAND → AIM
            # =========================

            if label == "Right":

                index_tip = hand_landmarks.landmark[8]
                index_base = hand_landmarks.landmark[5]

                tx = int(index_tip.x * width)
                ty = int(index_tip.y * height)

                bx = int(index_base.x * width)
                by = int(index_base.y * height)

                dx = tx - bx
                dy = ty - by

                cv2.arrowedLine(frame,(bx,by),(tx,ty),(0,255,0),3)

                if abs(dx) > abs(dy):

                    if dx > 20:
                        right_state = "AIM_RIGHT"
                    elif dx < -20:
                        right_state = "AIM_LEFT"

                else:

                    if dy > 20:
                        right_state = "AIM_DOWN"
                    elif dy < -20:
                        right_state = "AIM_UP"

    # =========================
    # SEND UDP
    # =========================

    send_udp()

    # =========================
    # DISPLAY
    # =========================

    cv2.putText(frame,f"L:{left_state}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.putText(frame,f"R:{right_state}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow("Hand Controller",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()