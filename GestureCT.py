import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# joystick
left_joystick_center = None

# gesture stability
last_action = ""
action_frames = 0
ACTION_HOLD = 5

# joystick settings
JOYSTICK_RADIUS = 180
DEADZONE = 60


# smoothing state per hand
hand_state = {
    "Left":{
        "cx":None,
        "cy":None,
        "radius":60
    },
    "Right":{
        "cx":None,
        "cy":None,
        "radius":60
    }
}


def smooth_value(old,new,alpha=0.7):
    return int(old*alpha + new*(1-alpha))


def get_palm_center(hand_landmarks,w,h):

    ids = [0,1,5,9,13,17]

    xs = [hand_landmarks.landmark[i].x for i in ids]
    ys = [hand_landmarks.landmark[i].y for i in ids]

    cx = int(np.mean(xs)*w)
    cy = int(np.mean(ys)*h)

    return cx,cy


def get_dynamic_radius(hand_landmarks,w,h):

    index_base = hand_landmarks.landmark[5]
    pinky_base = hand_landmarks.landmark[17]

    ix,iy = int(index_base.x*w),int(index_base.y*h)
    px,py = int(pinky_base.x*w),int(pinky_base.y*h)

    palm_width = np.sqrt((ix-px)**2 + (iy-py)**2)

    return int(palm_width*1.25)


def finger_state(hand_landmarks,center,radius,w,h):

    fingertip_ids = [4,8,12,16,20]

    cx,cy = center
    states = []

    for tip in fingertip_ids:

        x = int(hand_landmarks.landmark[tip].x*w)
        y = int(hand_landmarks.landmark[tip].y*h)

        dist = np.sqrt((x-cx)**2 + (y-cy)**2)

        states.append(0 if dist < radius else 1)

    return states


def stable_action(action):

    global last_action,action_frames

    if action == last_action:
        action_frames += 1
    else:
        action_frames = 0
        last_action = action

    if action_frames > ACTION_HOLD:
        return action

    return None


while True:

    ret,frame = cap.read()
    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for handLms,handType in zip(result.multi_hand_landmarks,
                                    result.multi_handedness):

            label = handType.classification[0].label

            cx,cy = get_palm_center(handLms,w,h)
            radius = get_dynamic_radius(handLms,w,h)

            state = hand_state[label]

            # smoothing per hand
            if state["cx"] is None:
                state["cx"],state["cy"] = cx,cy

            state["cx"] = smooth_value(state["cx"],cx)
            state["cy"] = smooth_value(state["cy"],cy)
            state["radius"] = smooth_value(state["radius"],radius)

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

                    cv2.line(frame,(jx-JOYSTICK_RADIUS,jy),
                                   (jx+JOYSTICK_RADIUS,jy),(255,255,0),1)

                    cv2.line(frame,(jx,jy-JOYSTICK_RADIUS),
                                   (jx,jy+JOYSTICK_RADIUS),(255,255,0),1)

                    dx = cx-jx
                    dy = cy-jy

                    distance = np.sqrt(dx**2 + dy**2)

                    action = "IDLE"

                    if distance < DEADZONE:
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
                        print(stable)

                    cv2.putText(frame,action,(30,50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,(0,255,255),2)

                else:
                    left_joystick_center = None

            # ---------------- RIGHT HAND ----------------

            if label == "Right":

                if fingers == [0,1,1,0,0]:

                    tip = handLms.landmark[8]
                    joint = handLms.landmark[6]

                    tx,ty = int(tip.x*w),int(tip.y*h)
                    jx,jy = int(joint.x*w),int(joint.y*h)

                    dx = tx-jx
                    dy = ty-jy

                    cv2.arrowedLine(frame,(jx,jy),(tx,ty),(0,0,255),3)

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
                        print(stable)

                    cv2.putText(frame,action,(30,90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,(255,0,255),2)

                elif fingers == [0,0,0,0,0]:

                    stable = stable_action("ATTACK")
                    if stable:
                        print("ATTACK")

                elif fingers == [1,1,1,1,1]:

                    stable = stable_action("DEFEND")
                    if stable:
                        print("DEFEND")

            mp_draw.draw_landmarks(frame,handLms,mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Control Stable",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()