import cv2
import mediapipe as mp
import time
import math


class HandDetector():
    def __init__(self, mode=False, max_hands=2, det_conf=0.5, trac_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.det_conf = det_conf
        self.trac_conf = trac_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.det_conf,
            min_tracking_confidence=self.trac_conf
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=False):
        """Find hands in the image and optionally draw landmarks"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=False):
        """Find the position of hand landmarks"""
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:  # Check if the requested hand exists
                my_hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 200, 100), cv2.FILLED)
        return lm_list

    def detect_pinch(self, img, lm_list):
        """Detect pinch gesture between thumb and index finger"""
        cx, cy = 0, 0
        pinch = False

        if len(lm_list) >= 9:  # Ensure we have at least index finger and thumb landmarks
            try:
                # Thumb tip
                x1, y1 = lm_list[4][1], lm_list[4][2]
                # Index finger tip
                x2, y2 = lm_list[8][1], lm_list[8][2]

                # Center point between the two fingertips
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw circles at the fingertips
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Calculate distance between fingertips
                length = math.hypot(x2 - x1, y2 - y1)

                # Determine if pinch gesture is detected
                if length < 40:  # Reduced threshold for more precise pinch
                    cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                    pinch = True

                else:
                    pinch = False
            except IndexError:
                pass  # Handle case where landmarks are not properly detected

        return cx, cy, pinch

    def detect_gesture(self, img, lm_list):
        """Detect different hand gestures for brush selection"""
        gesture = None

        if len(lm_list) >= 21:  # We need all fingers for gesture detection
            try:
                # Fingertips y-coordinates
                thumb_tip_y = lm_list[4][2]
                index_tip_y = lm_list[8][2]
                middle_tip_y = lm_list[12][2]
                ring_tip_y = lm_list[16][2]
                pinky_tip_y = lm_list[20][2]

                # Finger base y-coordinates (for comparison)
                index_base_y = lm_list[5][2]
                middle_base_y = lm_list[9][2]
                ring_base_y = lm_list[13][2]
                pinky_base_y = lm_list[17][2]

                # Check if fingers are up or down
                index_up = index_tip_y < index_base_y
                middle_up = middle_tip_y < middle_base_y
                ring_up = ring_tip_y < ring_base_y
                pinky_up = pinky_tip_y < pinky_base_y

                # Different gestures for different brush options
                if index_up and not middle_up and not ring_up and not pinky_up:
                    gesture = "index"  # Index finger only
                elif index_up and middle_up and not ring_up and not pinky_up:
                    gesture = "index and middle"  # Index and middle fingers
                elif index_up and middle_up and ring_up and not pinky_up:
                    gesture = "Three fingers"  # Three fingers
                elif index_up and middle_up and ring_up and pinky_up:
                    gesture = "All up"  # All fingers up
                elif not index_up and not middle_up and not ring_up and pinky_up:
                    gesture = "pinky"  # Only pinky up

                # Display the detected gesture
                if gesture:
                    cv2.putText(img, f"Gesture: {gesture}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            except IndexError:
                pass

        return gesture


def main():
    """Test function for the hand detector"""
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from camera")
            break

        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if lm_list:
            cx, cy, pinch = detector.detect_pinch(img, lm_list)
            gesture = detector.detect_gesture(img, lm_list)

            if pinch:
                cv2.putText(img, "Pinch Detected", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if gesture:
                cv2.putText(img, f"Gesture: {gesture}", (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Calculate and display FPS
        cv2.imshow('Hand Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
