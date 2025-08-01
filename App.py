import sys
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from UI import Ui_MainWindow

# --- Load emotion model ---
model = load_model('custom_cnn_fer.h5')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# --- Initialize MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Emotion weights for understanding score
EMOTION_WEIGHTS = {
    'Happy': 1.3,
    'Surprise': 1.2,
    'Neutral': 1.1,
    'Sad': 0.8,
    'Angry': 0.7,
    'Fear': 0.8,
    'Disgust': 0.7,
    'Unknown': 0.6
}


class FocusAnalysisApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Connect buttons
        self.start_btn.clicked.connect(self.start_analysis)
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.report_btn.clicked.connect(self.generate_report)

        # Initialize variables
        self.is_analyzing = False
        self.start_time = 0
        self.cap = None
        self.face_mesh = None
        self.face_detection = None
        self.focus_levels = []
        self.timestamps = []
        self.understanding_score = 50
        self.understanding_scores = [50]
        self.understanding_timestamps = [0]
        self.last_understanding_update = 0
        self.delay_between_understanding_update = 1.5
        self.smoothing_window = 15
        self.focus_history = deque(maxlen=self.smoothing_window)

        # Setup timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Set initial status
        self.status_label.setText("Sẵn sàng bắt đầu theo dõi")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

    def start_analysis(self):
        if not self.is_analyzing:
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.show_error("Không thể mở camera!")
                return

            # Initialize face detection
            self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)
            self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

            # Reset variables
            self.start_time = time.time()
            self.focus_levels = []
            self.timestamps = []
            self.understanding_score = 50
            self.understanding_scores = [50]
            self.understanding_timestamps = [0]
            self.last_understanding_update = self.start_time
            self.focus_history = deque(maxlen=self.smoothing_window)

            # Update UI
            self.is_analyzing = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Đang theo dõi...")
            self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
            self.timer.start(30)  # Update every 30ms

    def stop_analysis(self):
        if self.is_analyzing:
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None

            self.is_analyzing = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Đã dừng theo dõi")
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold;")

            # Show final frame
            black_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.display_image(black_image)

    def estimate_pose(self, landmarks, img_w, img_h):
        indices = [33, 263, 1, 61, 291, 199]
        pts_2d = [[landmarks[idx].x * img_w, landmarks[idx].y * img_h] for idx in indices]

        model_points = np.array([
            [-30, 0, 0],
            [30, 0, 0],
            [0, 0, 0],
            [-25, -30, 0],
            [25, -30, 0],
            [0, -65, 0],
        ], dtype=np.float32)

        image_points = np.array(pts_2d, dtype=np.float32)
        focal_length = img_w
        camera_matrix = np.array([[focal_length, 0, img_w / 2],
                                  [0, focal_length, img_h / 2],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles  # pitch, yaw, roll

    def head_orientation_factor(self, yaw):
        abs_yaw = abs(yaw)
        if abs_yaw <= 40:
            return 1.0
        elif abs_yaw >= 80:
            return 0.6
        else:
            return 0.8

    def emotion_score(self, emotion_label, confidence):
        if emotion_label == "Unknown":
            return 0.0, -2

        weight = EMOTION_WEIGHTS.get(emotion_label, 1.0)
        focus_multiplier = confidence * weight

        if emotion_label in ['Happy', 'Surprise']:
            return focus_multiplier, 1
        elif emotion_label == 'Neutral':
            return focus_multiplier, 0
        else:
            return focus_multiplier, -1

    def calculate_smoothed_focus(self):
        if not self.focus_history:
            return 0
        return sum(self.focus_history) / len(self.focus_history)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        emotion_label = "Unknown"
        confidence = 0.0
        face_roi = None

        detection_results = self.face_detection.process(rgb)
        if detection_results.detections:
            # Get the largest face
            largest_face = max(detection_results.detections,
                               key=lambda det: det.location_data.relative_bounding_box.width *
                                               det.location_data.relative_bounding_box.height)

            bbox = largest_face.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Ensure coordinates are within frame bounds
            x, y, width, height = max(0, x), max(0, y), min(w, width), min(h, height)
            face_roi = frame[y:y + height, x:x + width]

            if face_roi.size > 0:
                # Preprocess for emotion recognition
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray_roi, (48, 48))
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result_predict = model.predict(reshaped, verbose=0)

                label = np.argmax(result_predict)
                confidence = np.max(result_predict)
                emotion_label = labels_dict[label]

        # Head pose estimation
        focus = 0
        mesh_results = self.face_mesh.process(rgb)
        if mesh_results.multi_face_landmarks and face_roi is not None:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            pitch, yaw, roll = self.estimate_pose(landmarks, w, h)

            # Calculate focus score
            focus_multiplier, understanding_delta = self.emotion_score(emotion_label, confidence)
            orientation_factor = self.head_orientation_factor(yaw)
            focus = 100 * focus_multiplier * orientation_factor
            focus = max(0, min(100, focus))

        # Update focus history
        self.focus_history.append(focus)
        smoothed_focus = self.calculate_smoothed_focus()
        self.focus_levels.append(smoothed_focus)
        self.timestamps.append(time.time() - self.start_time)

        # Update understanding score
        current_time = time.time()
        if current_time - self.last_understanding_update > self.delay_between_understanding_update:
            if smoothed_focus < 50:
                self.understanding_score -= 1
            else:
                _, base_delta = self.emotion_score(emotion_label, confidence)

                if smoothed_focus >= 70:
                    self.understanding_score += base_delta + 1
                elif smoothed_focus >= 85:
                    self.understanding_score += base_delta + 2
                else:
                    self.understanding_score += base_delta

            self.understanding_score = max(0, min(100, self.understanding_score))
            self.understanding_scores.append(self.understanding_score)
            self.understanding_timestamps.append(current_time - self.start_time)
            self.last_understanding_update = current_time

        # Update UI
        self.update_ui(frame, emotion_label, confidence, smoothed_focus, current_time)

    def update_ui(self, frame, emotion_label, confidence, smoothed_focus, current_time):
        # Update camera display
        self.display_image(frame)

        # Update gauges
        self.focus_gauge.setValue(smoothed_focus)
        self.understanding_gauge.setValue(self.understanding_score)

        # Update emotion labels
        self.emotion_label.setText(emotion_label)
        self.confidence_label.setText(f"Độ tin cậy: {confidence * 100:.1f}%")

        # Update time
        elapsed = current_time - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self.time_label.setText(f"Thời gian: {mins:02d}:{secs:02d}")

        # Update warning
        if smoothed_focus < 50:
            self.warning_label.setText("CẢNH BÁO: Mức độ tập trung thấp!")
        else:
            self.warning_label.setText("")

        # Update emotion color
        color = "#2196F3"  # Default blue
        if emotion_label == "Happy":
            color = "#FFC107"  # Yellow
        elif emotion_label == "Angry":
            color = "#F44336"  # Red
        elif emotion_label == "Sad":
            color = "#2196F3"  # Blue
        elif emotion_label == "Surprise":
            color = "#9C27B0"  # Purple

        self.emotion_label.setStyleSheet(f"color: {color};")

        # Update status
        if smoothed_focus >= 70:
            self.status_label.setText("Tập trung tốt")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif smoothed_focus >= 50:
            self.status_label.setText("Tập trung trung bình")
            self.status_label.setStyleSheet("color: #FFC107; font-weight: bold;")
        else:
            self.status_label.setText("Tập trung thấp")
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold;")

    def display_image(self, frame):
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # Create QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale and display
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.KeepAspectRatio
        ))

    def generate_report(self):
        if not self.focus_levels:
            self.show_error("Không có dữ liệu để xuất báo cáo!")
            return

        # Calculate statistics
        total_time = self.timestamps[-1] if self.timestamps else 1
        avg_focus = np.mean(self.focus_levels) if self.focus_levels else 0
        min_focus = np.min(self.focus_levels) if self.focus_levels else 0
        max_focus = np.max(self.focus_levels) if self.focus_levels else 0

        # Calculate time at different focus levels
        low_focus_time = sum(1 for f in self.focus_levels if f < 50) * (total_time / len(self.focus_levels))
        medium_focus_time = sum(1 for f in self.focus_levels if 50 <= f < 80) * (total_time / len(self.focus_levels))
        high_focus_time = sum(1 for f in self.focus_levels if f >= 80) * (total_time / len(self.focus_levels))

        # Calculate percentages
        low_percent = (low_focus_time / total_time) * 100
        medium_percent = (medium_focus_time / total_time) * 100
        high_percent = (high_focus_time / total_time) * 100

        # Performance rating
        if avg_focus >= 70 and self.understanding_score >= 70:
            performance_rating = "XUẤT SẮC"
            feedback = "Bạn đã duy trì sự tập trung và hiểu bài xuất sắc!"
        elif avg_focus >= 60 and self.understanding_score >= 60:
            performance_rating = "TỐT"
            feedback = "Kết quả tốt, hãy tiếp tục phát huy!"
        elif avg_focus >= 50 and self.understanding_score >= 50:
            performance_rating = "TRUNG BÌNH"
            feedback = "Kết quả trung bình, cần cải thiện sự tập trung"
        else:
            performance_rating = "CẦN CẢI THIỆN"
            feedback = "Hãy xem lại phương pháp học tập để cải thiện kết quả"

        # Create report text
        report_text = (
            f"===== BÁO CÁO BUỔI HỌC =====\n\n"
            f"⏱ Thời lượng: {int(total_time // 60)} phút {int(total_time % 60)} giây\n"
            f"🎯 Tập trung trung bình: {avg_focus:.1f}%\n"
            f"📊 Hiểu bài cuối cùng: {self.understanding_score}%\n\n"
            f"📈 PHÂN BỐ MỨC TẬP TRUNG:\n"
            f"  • Thấp (<50%): {low_percent:.1f}%\n"
            f"  • Trung bình (50-80%): {medium_percent:.1f}%\n"
            f"  • Cao (>80%): {high_percent:.1f}%\n\n"
            f"🏆 ĐÁNH GIÁ HIỆU SUẤT: {performance_rating}\n"
            f"💬 Nhận xét: {feedback}\n\n"
            f"============================="
        )

        # Show report in a message box
        msg = QMessageBox()
        msg.setWindowTitle("Báo cáo buổi học")
        msg.setText(report_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def show_error(self, message):
        QMessageBox.critical(self, "Lỗi", message)

    def closeEvent(self, event):
        self.stop_analysis()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FocusAnalysisApp()
    window.show()
    sys.exit(app.exec_())