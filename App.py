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

# --- Khởi tạo MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Gắn mức độ hiểu cho cảm xúc
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

# Lớp chính kế thừa từ QMainWindow và file thiết kế giao diện UI
class FocusAnalysisApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)   # Thiết lập giao diện từ file Qt Designer

        # Kết nối button
        self.start_btn.clicked.connect(self.start_analysis)
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.report_btn.clicked.connect(self.generate_report)

        # Initialize variables
        # Khởi tạo các biến theo dõi trạng thái và dữ liệu phân tích
        self.is_analyzing = False  # Cờ cho biết có đang phân tích không
        self.start_time = 0  # Thời điểm bắt đầu phân tích
        self.cap = None  # Đối tượng webcam
        self.face_mesh = None  # Dùng để lấy điểm landmark khuôn mặt (Face Mesh)
        self.face_detection = None  # Dùng để phát hiện khuôn mặt
        self.focus_levels = []  # Danh sách điểm tập trung theo thời gian
        self.timestamps = []  # Mốc thời gian tương ứng với mỗi điểm
        self.understanding_score = 50  # Điểm hiểu bài ban đầu (trung bình)
        self.understanding_scores = [50]  # Lịch sử điểm hiểu bài
        self.understanding_timestamps = [0]  # Thời điểm ghi nhận điểm hiểu bài
        self.last_understanding_update = 0  # Lần cuối cập nhật điểm hiểu bài
        self.delay_between_understanding_update = 1.5  # Giãn cách tối thiểu giữa các lần cập nhật điểm hiểu bài
        self.smoothing_window = 15  # Số lượng điểm để làm mượt dữ liệu
        self.focus_history = deque(maxlen=self.smoothing_window)  # Hàng đợi chứa điểm tập trung để làm mượt

        # Setup timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame) # Gọi hàm update_frame mỗi lần timeout

        # Giao diện trạng thái ban đầu
        self.status_label.setText("Sẵn sàng bắt đầu theo dõi")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

    #Bắt đầu phân tích
    def start_analysis(self):
        if not self.is_analyzing:
            # Initialize video capture
            self.cap = cv2.VideoCapture(0) # Mở webcam (camera 0)
            if not self.cap.isOpened():
                self.show_error("Không thể mở camera!")
                return

            # Khởi tạo mô-đun Mediapipe
            self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)
            self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
            #face_mesh: theo dõi 468 điểm landmark trên khuôn mặt.
            # face_detection: xác định khuôn mặt trong khung hình.
            # Độ tin cậy tối thiểu là 50%.

            # Reset dữ liệu theo dõi
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
            self.start_btn.setEnabled(False) #tắt
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Đang theo dõi...")
            self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
            self.timer.start(30)   # Gọi update_frame mỗi 30ms

    def stop_analysis(self):
        if self.is_analyzing:
            self.timer.stop() #dừng cam
            if self.cap:
                self.cap.release()
                self.cap = None

            self.is_analyzing = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Đã dừng theo dõi")
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold;") #màu đỏ

            # Hiển thị khung hình đen khi dừng
            black_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.display_image(black_image) #hiện khung lên giao diện

    #ước lượng góc đầu
    def estimate_pose(self, landmarks, img_w, img_h):
        indices = [33, 263, 1, 61, 291, 199] #Chọn 6 điểm landmark quan trọng (mắt, mũi, miệng) để định vị tư thế đầu
        # Chuyển đổi sang tọa độ pixel trên ảnh (2D)
        pts_2d = [[landmarks[idx].x * img_w, landmarks[idx].y * img_h] for idx in indices]

    #Tập các điểm 3D giả định (giống cấu trúc khuôn mặt trung bình) trong không gian thực.
        model_points = np.array([
            [-30, 0, 0],
            [30, 0, 0],
            [0, 0, 0],
            [-25, -30, 0],
            [25, -30, 0],
            [0, -65, 0],
        ], dtype=np.float32)

        # Chuyển đổi các điểm ảnh sang float32 để dùng với solvePnP
        image_points = np.array(pts_2d, dtype=np.float32)
        focal_length = img_w

        # Ma trận nội tại camera (camera matrix) giả định đơn giản
        camera_matrix = np.array([[focal_length, 0, img_w / 2],
                                  [0, focal_length, img_h / 2],
                                  [0, 0, 1]], dtype="double")
        # Không dùng méo ống kính (giả định camera lý tưởng)
        dist_coeffs = np.zeros((4, 1))

        # Dự đoán góc quay (rotation vector) dùng solvePnP
        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        # Chuyển vector quay sang ma trận quay (rotation matrix)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        # Tính toán góc pitch, yaw, roll từ rotation matrix
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles  # pitch, yaw, roll

    #Điểm tập trung của các hướng đầu
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
    #tính điểm dựa vào cảm xúc
        if emotion_label in ['Happy', 'Surprise']:
            return focus_multiplier, 1
        elif emotion_label == 'Neutral':
            return focus_multiplier, 0
        else:
            return focus_multiplier, -1 #các cảm xúc tiêu cực

    #Trung bình hóa điểm tập trung trong cửa sổ trượt
    def calculate_smoothed_focus(self):
        if not self.focus_history:
            return 0
        return sum(self.focus_history) / len(self.focus_history)

    def update_frame(self): #update frame cho từng khung ảnh
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read() # Đọc 1 khung hình từ webcam
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #chuyển ảnh sang rgb

        # ===== Phân phát hiện khuôn mat =========

        #biến khởi tạo
        emotion_label = "Unknown"
        confidence = 0.0
        face_roi = None

        #dùng mediapipe face detection để tìm ảnh khuôn mặt xám
        detection_results = self.face_detection.process(rgb)
        if detection_results.detections:
            # Chọn khuôn mặt lớn nhất
            largest_face = max(detection_results.detections,
                               key=lambda det: det.location_data.relative_bounding_box.width *
                                               det.location_data.relative_bounding_box.height)

            # Lấy tọa độ bounding box tương đối rồi chuyển sang tọa độ tuyệt đối (pixel)
            bbox = largest_face.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            # Đảm bảo không vượt quá kích thước ảnh
            x, y, width, height = max(0, x), max(0, y), min(w, width), min(h, height)
            face_roi = frame[y:y + height, x:x + width]

            if face_roi.size > 0:
                #=========Phần nhận diện cảm xúc===========
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray_roi, (48, 48))
                normalize = resized / 255.0 # Chuẩn hóa giá trị pixel về [0, 1]
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result_predict = model.predict(reshaped, verbose=0)

                label = np.argmax(result_predict) # Dự đoán cảm xúc
                confidence = np.max(result_predict)
                emotion_label = labels_dict[label]
                #Lấy nhãn cảm xúc dự đoán và độ tin cậy cao nhất

        # Head pose estimation
        focus = 0
        mesh_results = self.face_mesh.process(rgb)  #chứa kết quả nhận diện mesh (lưới điểm) khuôn mặt từ ảnh RGB
        if mesh_results.multi_face_landmarks and face_roi is not None:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            pitch, yaw, roll = self.estimate_pose(landmarks, w, h) #Trích xuất landmark khuôn mặt đầu tiên để ước lượng góc xoay đầu

            # Tính điểm tập trung
            focus_multiplier, understanding_delta = self.emotion_score(emotion_label, confidence)
            orientation_factor = self.head_orientation_factor(yaw) #biến
            focus = 100 * focus_multiplier * orientation_factor #Điểm tập trung (focus) = cảm xúc × hướng đầu.
            focus = max(0, min(100, focus))

        # ============Cập nhật mức độ tập trung=========
        self.focus_history.append(focus)  #Lưu lại điểm tập trung hiện tại.
        smoothed_focus = self.calculate_smoothed_focus() #Dùng trung bình trượt để làm mượt mức tập trung
        #Ghi lại lịch sử điểm tập trung và mốc thời gian.
        self.focus_levels.append(smoothed_focus)
        self.timestamps.append(time.time() - self.start_time)

        # ==========Cập nhật mức độ hiểu bài===========
        current_time = time.time()
        #Cập nhật hiểu bài nếu đủ thời gian trôi qua từ lần cập nhật trước
        if current_time - self.last_understanding_update > self.delay_between_understanding_update:
            if smoothed_focus < 50:
                self.understanding_score -= 1 #Nếu tập trung thấp, giảm điểm hiểu bài.
            else:
                _, base_delta = self.emotion_score(emotion_label, confidence) #Nếu tập trung >= 50, tính mức tăng điểm hiểu bài từ cảm xúc

                #Nếu tập trung cao, tăng điểm hiểu bài nhiều hơn.
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

        # =======Cập nhật lại UI========
        self.update_ui(frame, emotion_label, confidence, smoothed_focus, current_time)

    def update_ui(self, frame, emotion_label, confidence, smoothed_focus, current_time):
        # Hiện thị hình ảnh webcam
        self.display_image(frame)

        # Cập nhật đồng hồ đo mức tập trung và hiểu bài.
        self.focus_gauge.setValue(smoothed_focus)
        self.understanding_gauge.setValue(self.understanding_score)

        # Cập nhật nhãn cảm xúc và độ tin cậy.
        self.emotion_label.setText(emotion_label)
        self.confidence_label.setText(f"Độ tin cậy: {confidence * 100:.1f}%")

        #Hiển thị thời gian đã học
        elapsed = current_time - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self.time_label.setText(f"Thời gian: {mins:02d}:{secs:02d}")

        # Cảnh báo nếu người học không tập trung.
        if smoothed_focus < 50:
            self.warning_label.setText("CẢNH BÁO: Mức độ tập trung thấp!")
        else:
            self.warning_label.setText("")

        # Cập nhật màu
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

        # Cập nhật trạng thái
        if smoothed_focus >= 70:
            self.status_label.setText("Tập trung tốt")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif smoothed_focus >= 50:
            self.status_label.setText("Tập trung trung bình")
            self.status_label.setStyleSheet("color: #FFC107; font-weight: bold;")
        else:
            self.status_label.setText("Tập trung thấp")
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold;")

    #Chuyển ảnh từ OpenCV sang định dạng QPixmap và hiển thị lên giao diện PyQt.
    def display_image(self, frame):
        # Chuyển sang RGB
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
    #=====Tạo báo cáo=======
    def generate_report(self):
        if not self.focus_levels:
            self.show_error("Không có dữ liệu để xuất báo cáo!")
            return

        # Calculate statistics
        total_time = self.timestamps[-1] if self.timestamps else 1
        avg_focus = np.mean(self.focus_levels) if self.focus_levels else 0
        min_focus = np.min(self.focus_levels) if self.focus_levels else 0
        max_focus = np.max(self.focus_levels) if self.focus_levels else 0

        #Tính thời gian học, độ tập trung trung bình, min/max
        low_focus_time = sum(1 for f in self.focus_levels if f < 50) * (total_time / len(self.focus_levels))
        medium_focus_time = sum(1 for f in self.focus_levels if 50 <= f < 80) * (total_time / len(self.focus_levels))
        high_focus_time = sum(1 for f in self.focus_levels if f >= 80) * (total_time / len(self.focus_levels))

        # Thời gian phân bố theo các mức độ tập trung.
        low_percent = (low_focus_time / total_time) * 100
        medium_percent = (medium_focus_time / total_time) * 100
        high_percent = (high_focus_time / total_time) * 100

        # Chuyển sang phần trăm để trực quan hóa.
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

    #===== Chạy ứng dụng=========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FocusAnalysisApp()
    window.show()
    sys.exit(app.exec_())