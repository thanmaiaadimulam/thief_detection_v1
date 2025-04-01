import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        # Use named parameters instead of positional parameters
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,  # This replaces upBody parameter in newer versions
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS
            )

        return img

    def findPosition(self, img, draw=False):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    #def in_or_out(self,lmList,protected_zone):
        #for id,cx,cy in lmList:
         #   print(id,cx,cy)
          #  if cx<protected_zone[0][0] or cx>protected_zone[1][0] and cx<protected_zone[0][1] or cx>protected_zone[1][1]:
         #       intrution = False
         #   if cx>protected_zone[0][0] or cx<protected_zone[1][0] and cx>protected_zone[0][1] or cx<protected_zone[1][1]:
         #       intrution = True
         #   return intrution



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        #intrution = detector.in_or_out(lmList,protected_zone=[(0,0),(0,0)])

        if len(lmList) != 0:
            # Example: Print and highlight the position of nose (landmark 0)
            cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (0, 0, 255), cv2.FILLED)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
