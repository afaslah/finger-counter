import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FINGER_TIPS = [8, 12, 16, 20]
FINGER_JOINTS = [6, 10, 14, 18]
THUMB_TIP_ID = 4
THUMB_JOINT_ID = 3
FIST_DISTANCE_THRESHOLD = 0.09
prev_time = 0

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def is_thumb_extended(hand_landmarks):
    cmc = hand_landmarks.landmark[1]
    mcp = hand_landmarks.landmark[2]
    tip = hand_landmarks.landmark[4]

    # Vectors
    vec1 = [mcp.x - cmc.x, mcp.y - cmc.y]
    vec2 = [tip.x - mcp.x, tip.y - mcp.y]

    # Angle between vectors
    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    mag1 = math.hypot(*vec1)
    mag2 = math.hypot(*vec2)
    angle = math.degrees(math.acos(dot / (mag1 * mag2 + 1e-6)))
    return angle < 50  # Thumb extended

#fitsbump
def is_fist(hand_landmarks, wrist):
    for tip_id in FINGER_TIPS + [THUMB_TIP_ID]:
        if distance(hand_landmarks.landmark[tip_id], wrist) > FIST_DISTANCE_THRESHOLD:
            return False
    return True

def count_fingers(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    count = 0

    # Count fingers
    for tip_id, joint_id in zip(FINGER_TIPS, FINGER_JOINTS):
        tip = hand_landmarks.landmark[tip_id]
        joint = hand_landmarks.landmark[joint_id]
        if tip.y < joint.y and distance(tip, wrist) > 0.1:
            count += 1

    # Thumb
    if is_thumb_extended(hand_landmarks):
        count += 1

    # Fistbump check
    if is_fist(hand_landmarks, wrist):
        return 0

    return count

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            total_fingers += count_fingers(hand_landmarks)

            # line and dot in hand
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # Dot
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) # Line
            )

    # finger count
    cv2.putText(frame, f'Fingers: {total_fingers}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()