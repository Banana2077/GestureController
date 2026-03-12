import cv2
import mediapipe as mp
import numpy as np
import socket

# ---------------- UDP ----------------

UDP_IP = "127.0.0.1"
UDP_PORT = 5052
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

# ---------------- JOYSTICK ----------------

left_joystick_center = None
JOYSTICK_RADIUS = 220
DEADZONE = 55

# ---------------- ACTION ----------------

left_action = "IDLE"
right_action = "NONE"

# ---------------- DETECT FLAG ----------------

hand_detected_left = False
hand_detected_right = False

# ---------------- STABILITY ----------------

last_action = ""
action_frames = 0
ACTION_HOLD = 4

# ---------------- HAND STATE ----------------

hand_state = {
    "Left": {"cx": None, "cy": None, "radius": 60},
    "Right": {"cx": None, "cy": None, "radius": 60}
}

# ---------------- FUNCTIONS ----------------

def smooth_value(old, new, alpha=0.45):
    return int(old * alpha + new * (1 - alpha))


def smooth_radius(old, new):
    return int(old * 0.2 + new * 0.8)


def get_palm_center(hand_landmarks, w, h):

    ids = [0,1,5,9,13,17]

    xs = [hand_landmarks.landmark[i].x for i in ids]
    ys = [hand_landmarks.landmark[i].y for i in ids]

    cx = int(np.mean(xs) * w)
    cy = int(np.mean(ys) * h)

    return cx, cy


def get_dynamic_radius(hand_landmarks, w, h):

    index_base = hand_landmarks.landmark[5]
    pinky_base = hand_landmarks.landmark[17]

    ix, iy = int(index_base.x*w), int(index_base.y*h)
    px, py = int(pinky_base.x*w), int(pinky_base.y*h)

    palm_width = np.sqrt((ix-px)**2 + (iy-py)**2)

    return int(palm_width * 1.3)


def finger_state(hand_landmarks, center, radius, w, h):

    fingertip_ids = [4,8,12,16,20]

    cx, cy = center
    states = []

    for tip in fingertip_ids:

        x = int(hand_landmarks.landmark[tip].x*w)
        y = int(hand_landmarks.landmark[tip].y*h)

        dist = np.sqrt((x-cx)**2 + (y-cy)**2)

        states.append(0 if dist < radius else 1)

    return states


def stable_action(action):

    global last_action, action_frames

    if action == last_action:
        action_frames += 1
    else:
        last_action = action
        action_frames = 0

    if action_frames > ACTION_HOLD:
        return action

    return None


# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected_left = False
    hand_detected_right = False

    if result.multi_hand_landmarks:

        for handLms, handType in zip(result.multi_hand_landmarks,
                                     result.multi_handedness):

            label = handType.classification[0].label

            if label == "Left":
                hand_detected_left = True
            if label == "Right":
                hand_detected_right = True

            cx, cy = get_palm_center(handLms, w, h)
            radius = get_dynamic_radius(handLms, w, h)

            state = hand_state[label]

            if state["cx"] is None:
                state["cx"], state["cy"] = cx, cy

            state["cx"] = smooth_value(state["cx"], cx)
            state["cy"] = smooth_value(state["cy"], cy)
            state["radius"] = smooth_radius(state["radius"], radius)

            cx = state["cx"]
            cy = state["cy"]
            radius = state["radius"]

            cv2.circle(frame,(cx,cy),radius,(0,255,0),2)

            fingers = finger_state(handLms,(cx,cy),radius,w,h)

            # ---------------- LEFT HAND ----------------

            if label == "Left":

                if fingers == [0,0,0,0,0]:

                    if left_joystick_center is None:
                        left_joystick_center = (cx,cy)

                    jx,jy = left_joystick_center

                    cv2.circle(frame,(jx,jy),JOYSTICK_RADIUS,(255,255,0),2)
                    cv2.circle(frame,(jx,jy),DEADZONE,(100,100,255),2)

                    dx = cx - jx
                    dy = cy - jy

                    dist = np.sqrt(dx**2 + dy**2)

                    action = "IDLE"

                    if dist < DEADZONE:
                        action = "MOVE_FORWARD"

                    else:

                        if abs(dx) > abs(dy):

                            if dx < 0:
                                action = "MOVE_LEFT"
                            else:
                                action = "MOVE_RIGHT"

                        else:

                            if dy < 0:
                                action = "JUMP"
                            else:
                                action = "CROUCH"

                    stable = stable_action(action)

                    if stable:
                        left_action = stable

                    cv2.putText(frame,left_action,(30,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

                else:
                    left_action = "IDLE"
                    left_joystick_center = None

            # ---------------- RIGHT HAND ----------------

            if label == "Right":

                if fingers == [0,1,1,0,0]:

                    tip = handLms.landmark[8]
                    joint = handLms.landmark[6]

                    tx,ty = int(tip.x*w), int(tip.y*h)
                    jx,jy = int(joint.x*w), int(joint.y*h)

                    dx = tx - jx
                    dy = ty - jy

                    threshold = 15
                    action = "AIM_CENTER"

                    if abs(dx) > abs(dy):

                        if dx > threshold:
                            action = "AIM_RIGHT"
                        elif dx < -threshold:
                            action = "AIM_LEFT"

                    else:

                        if dy > threshold:
                            action = "AIM_DOWN"
                        elif dy < -threshold:
                            action = "AIM_UP"

                    stable = stable_action(action)

                    if stable:
                        right_action = stable

                elif fingers == [0,0,0,0,0]:

                    right_action = "ATTACK"

                elif fingers == [1,1,1,1,1]:

                    right_action = "DEFEND"

            mp_draw.draw_landmarks(frame,handLms,mp_hands.HAND_CONNECTIONS)

    # ---------------- RESET WHEN HAND LOST ----------------

    if not hand_detected_left:
        left_action = "IDLE"
        left_joystick_center = None

    if not hand_detected_right:
        right_action = "NONE"

    # ---------------- SEND TO UNITY ----------------

    data = f"L:{left_action}|R:{right_action}"
    sock.sendto(data.encode(),(UDP_IP,UDP_PORT))

    cv2.imshow("Hand Control",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()