import cv2
import tracking_module as htm
import pose_module as pose
import os
import time
import hashlib
import numpy as np
from datetime import datetime


class ZoneSecuritySystem:
    def __init__(self):
        self.pinch_list = []
        self.cap = None
        self.detector = None
        self.pose_detector = None
        self.password_hash = None  # Will store a hashed password
        self.system_state = "INACTIVE"  # Possible states: INACTIVE, BUTTON_PRESSED, PASSWORD_VERIFY, SETTING_ZONE, MONITORING, OFFSET_ZONE
        self.zone_coords = [1, 1, 3, 3]  # Default zone: [x1, y1, x2, y2]
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.save_directory = "security_captures"

        # Mouse interaction variables
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Initialize password (in a real system, this would be loaded from a secure storage)
        self.initialize_password()

    def initialize_password(self):
        # Default password hash (for "admin123")
        # In a real application, this would be securely stored and loaded
        self.password_hash = hashlib.sha256("admin123".encode()).hexdigest()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, input_password):
        hashed_input = self.hash_password(input_password)
        return hashed_input == self.password_hash

    def prompt_for_password(self):
        # Create an OpenCV-based password input dialog
        password = ""
        entering_password = True

        # Create a blank image for the password input
        password_img = np.zeros((200, 400, 3), dtype=np.uint8)

        while entering_password:
            # Update the password display
            password_img.fill(0)  # Clear the image

            # Draw text
            cv2.putText(password_img, "Enter Password:", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the password as asterisks
            masked_pwd = '*' * len(password)
            cv2.putText(password_img, masked_pwd, (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(password_img, "Press Enter to submit, Esc to cancel", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Show the password input window
            cv2.imshow("Password Input", password_img)

            # Get key press
            key = cv2.waitKey(0) & 0xFF

            # Handle key press
            if key == 27:  # Esc key
                entering_password = False
                cv2.destroyWindow("Password Input")
                return False
            elif key == 13:  # Enter key
                entering_password = False
                cv2.destroyWindow("Password Input")

                # Verify password
                if self.verify_password(password):
                    # Show success message
                    success_img = np.zeros((200, 400, 3), dtype=np.uint8)
                    cv2.putText(success_img, "Password Verified!", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Success", success_img)
                    cv2.waitKey(1000)
                    cv2.destroyWindow("Success")
                    return True
                else:
                    # Show error message
                    error_img = np.zeros((200, 400, 3), dtype=np.uint8)
                    cv2.putText(error_img, "Incorrect Password!", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Error", error_img)
                    cv2.waitKey(1500)
                    cv2.destroyWindow("Error")
                    return False
            elif key == 8:  # Backspace
                password = password[:-1] if password else ""
            elif 32 <= key <= 126:  # Printable ASCII characters
                password += chr(key)

    def mouse_callback(self, event, x, y, flags, param):
        # Handle mouse events for zone setting
        if self.system_state in ["SETTING_ZONE", "OFFSET_ZONE"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_x, self.start_y = x, y
                self.end_x, self.end_y = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.end_x, self.end_y = x, y

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.end_x, self.end_y = x, y

                # Update zone coordinates
                self.zone_coords = [
                    min(self.start_x, self.end_x),  # x_min
                    min(self.start_y, self.end_y),  # y_min
                    max(self.start_x, self.end_x),  # x_max
                    max(self.start_y, self.end_y)  # y_max
                ]

    def start_recording(self, img):
        # Get current frame dimensions
        height, width, _ = img.shape

        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_directory, f"intrusion_{timestamp}.avi")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        self.recording = True
        self.recording_start_time = time.time()
        print(f"Recording started: {filename}")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Recording stopped")

    def capture_image(self, img):
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_directory, f"intrusion_image_{timestamp}.jpg")

        # Save the image
        cv2.imwrite(filename, img)
        print(f"Image captured: {filename}")

    def handle_button_press(self, key):
        # Simulate button presses with keyboard keys
        if self.system_state == "INACTIVE" and key == ord('b'):
            # Button pressed to start the system
            self.system_state = "PASSWORD_VERIFY"
            return True

        elif self.system_state == "OFFSET_ZONE" and key == ord('b'):
            # Button pressed to verify password for offset zone
            self.system_state = "PASSWORD_VERIFY_OFFSET"
            return True

        return False

    def update_system_state(self, key):
        # Handle state transitions based on button presses and other events

        # Handle button presses first
        if self.handle_button_press(key):
            return

        # Handle other state transitions
        if self.system_state == "PASSWORD_VERIFY":
            if self.prompt_for_password():
                self.system_state = "SETTING_ZONE"
                # Reset mouse coordinates when entering setting zone mode
                self.start_x, self.start_y = -1, -1
                self.end_x, self.end_y = -1, -1
            else:
                self.system_state = "INACTIVE"

        elif self.system_state == "PASSWORD_VERIFY_OFFSET":
            if self.prompt_for_password():
                self.system_state = "OFFSET_ZONE"
                # Reset mouse coordinates when entering offset zone mode
                self.start_x, self.start_y = -1, -1
                self.end_x, self.end_y = -1, -1
            else:
                self.system_state = "MONITORING"

        elif self.system_state == "SETTING_ZONE" and key == ord('s'):
            # Save the zone and start monitoring
            self.system_state = "MONITORING"

        elif self.system_state == "MONITORING" and key == ord('o'):
            # Request to offset/adjust the zone
            self.system_state = "OFFSET_ZONE"

        elif self.system_state == "OFFSET_ZONE" and key == ord('s'):
            # Save the adjusted zone and resume monitoring
            self.system_state = "MONITORING"

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Set width
        self.cap.set(4, 720)  # Set height
        self.detector = htm.HandDetector()
        self.pose_detector = pose.poseDetector()

        # Set up mouse callback
        cv2.namedWindow('Zone Security System')
        cv2.setMouseCallback('Zone Security System', self.mouse_callback)

        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture image from camera")
                break

            img = cv2.flip(img, 1)
            img = self.detector.find_hands(img)
            img = self.pose_detector.findPose(img)
            hand_lm_list = self.detector.find_position(img)
            body_lm_list = self.pose_detector.findPosition(img)

            # Display current system state
            cv2.putText(img, f"State: {self.system_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Instructions based on current state
            if self.system_state == "INACTIVE":
                cv2.putText(img, "Press 'b' to activate system", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            elif self.system_state == "SETTING_ZONE" or self.system_state == "OFFSET_ZONE":
                # Instructions for both gesture and mouse input
                cv2.putText(img, "Use pinch gesture or mouse drag to set zone. Press 's' to save", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Draw the rectangle being currently drawn with mouse if in drawing mode
                if self.drawing and self.start_x != -1 and self.start_y != -1:
                    cv2.rectangle(img,
                                  (self.start_x, self.start_y),
                                  (self.end_x, self.end_y),
                                  (255, 0, 0), 2)  # Blue for in-progress rectangle

                # Handle zone setting via pinch gesture
                if hand_lm_list:
                    gesture = self.detector.detect_gesture(img, hand_lm_list)
                    _, _, pinch = self.detector.detect_pinch(img, hand_lm_list)

                    if pinch and gesture == "All up":
                        x1, y1, pinch = self.detector.detect_pinch(img, hand_lm_list)
                        if len(self.pinch_list) <= 1:
                            self.pinch_list.append([x1, y1])
                        if len(self.pinch_list) > 1:
                            self.pinch_list[-1] = [x1, y1]

                    if len(self.pinch_list) > 0:
                        (x1, y1) = tuple(self.pinch_list[0])
                        if len(self.pinch_list) > 1:
                            (x2, y2) = tuple(self.pinch_list[-1])
                        else:
                            # Default second point if only one point has been set
                            x2, y2 = x1 + 100, y1 + 100

                        # Normalize coordinates
                        self.zone_coords = [
                            min(x1, x2),  # x_min
                            min(y1, y2),  # y_min
                            max(x1, x2),  # x_max
                            max(y1, y2)  # y_max
                        ]

            elif self.system_state == "MONITORING":
                cv2.putText(img, "Monitoring zone. Press 'o' to offset zone", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw the protected zone if coordinates are valid
            try:
                if all(coord >= 0 for coord in self.zone_coords):
                    cv2.rectangle(img,
                                  (self.zone_coords[0], self.zone_coords[1]),
                                  (self.zone_coords[2], self.zone_coords[3]),
                                  (0, 0, 255), 3)
            except Exception as e:
                print(f"Error drawing zone: {e}")
                # Reset zone coordinates if there's an error
                self.zone_coords = [100, 100, 300, 300]

            # Check for intrusions if in monitoring state
            if self.system_state == "MONITORING" and body_lm_list:
                intrusion = False

                try:
                    for id in body_lm_list:
                        if len(id) >= 3:  # Ensure we have valid coordinate data
                            cx, cy = id[1], id[2]
                            if (self.zone_coords[0] <= cx <= self.zone_coords[2]) and (
                                    self.zone_coords[1] <= cy <= self.zone_coords[3]):
                                intrusion = True
                                break  # One point inside is enough to trigger intrusion
                except Exception as e:
                    print(f"Error checking intrusion: {e}")

                # Display intrusion status
                if intrusion:
                    cv2.putText(img, "INTRUDER ALERT!", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Start recording if not already recording
                    if not self.recording:
                        self.start_recording(img)
                        # Capture a still image as well
                        self.capture_image(img)

                    # Write the current frame to the video
                    if self.recording and self.video_writer:
                        self.video_writer.write(img)

                        # Display recording indicator
                        cv2.circle(img, (30, 100), 10, (0, 0, 255), -1)  # Red recording dot

                        # Optional: Stop recording after a set duration (e.g., 10 seconds)
                        if time.time() - self.recording_start_time > 10:
                            self.stop_recording()
                else:
                    cv2.putText(img, "No intrusion detected", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Stop recording if no longer detecting intrusion
                    if self.recording:
                        self.stop_recording()

            cv2.imshow('Zone Security System', img)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            # Update system state based on key presses
            self.update_system_state(key)

            # Exit on 'q' key press
            if key == ord('q'):
                break

        # Clean up
        if self.recording:
            self.stop_recording()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    security_system = ZoneSecuritySystem()
    security_system.run()