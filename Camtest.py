import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
import matplotlib.pyplot as plt
from collections import deque

# --- Load emotion model ---
model = load_model('model_file.h5')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# --- Initialize MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

# --- Global variables ---
start_time = time.time()
focus_levels = []
timestamps = []
understanding_score = 50
understanding_scores = [50]  # Lưu lịch sử điểm hiểu bài
understanding_timestamps = [0]  # Thời điểm cập nhật
last_understanding_update = time.time()
delay_between_understanding_update = 1.5  # seconds
smoothing_window = 15  # Smoothing window size for focus
focus_history = deque(maxlen=smoothing_window)

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

def estimate_pose(landmarks, img_w, img_h):
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


def head_orientation_factor(yaw):
    abs_yaw = abs(yaw)
    if abs_yaw <= 40:
        return 1.0
    elif abs_yaw >= 80:
        return 0.6
    else:
        return 0.8

def emotion_score(emotion_label, confidence):
    """Returns focus multiplier and understanding delta"""
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


def calculate_smoothed_focus():
    """Calculate smoothed focus using moving average"""
    if not focus_history:
        return 0
    return sum(focus_history) / len(focus_history)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection with MediaPipe (better for rotated faces)
    emotion_label = "Unknown"
    confidence = 0.0
    face_roi = None

    detection_results = face_detection.process(rgb)
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
    mesh_results = face_mesh.process(rgb)
    if mesh_results.multi_face_landmarks and face_roi is not None:
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        pitch, yaw, roll = estimate_pose(landmarks, w, h)

        # Calculate focus score with emotion confidence and head orientation
        focus_multiplier, understanding_delta = emotion_score(emotion_label, confidence)
        orientation_factor = head_orientation_factor(yaw)
        focus = 100 * focus_multiplier * orientation_factor
        focus = max(0, min(100, focus))

    # Update focus history for smoothing
    focus_history.append(focus)
    smoothed_focus = calculate_smoothed_focus()
    focus_levels.append(smoothed_focus)
    timestamps.append(time.time() - start_time)

    # Update understanding score periodically
    current_time = time.time()
    if current_time - last_understanding_update > delay_between_understanding_update:
        if smoothed_focus < 50:
            understanding_score -= 1
        else:
            _, base_delta = emotion_score(emotion_label, confidence)

            # Tăng tốc khi tập trung cao
            if smoothed_focus >= 70:
                understanding_score += base_delta + 1  # +1 điểm bổ sung
            elif smoothed_focus >= 85:
                understanding_score += base_delta + 2  # +2 điểm khi tập trung rất cao
            else:
                understanding_score += base_delta

        understanding_score = max(0, min(100, understanding_score))
        # Lưu lịch sử hiểu bài
        understanding_scores.append(understanding_score)
        understanding_timestamps.append(current_time - start_time)

        last_understanding_update = current_time

    # Display information
    cv2.putText(frame, f"Emotion: {emotion_label} ({confidence:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Current Focus: {focus:.1f}%", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(frame, f"Elapsed Time: {int(current_time - start_time)}s", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
    cv2.putText(frame, f"Avg. Focus: {smoothed_focus:.1f}%", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Understanding: {understanding_score}%", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Warning for low focus
    if smoothed_focus < 50:
        cv2.putText(frame, "WARNING: Low Focus Level!", (20, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Focus Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Tính toán thống kê tổng kết ---
total_time = timestamps[-1] if timestamps else 1
avg_focus = np.mean(focus_levels) if focus_levels else 0
min_focus = np.min(focus_levels) if focus_levels else 0
max_focus = np.max(focus_levels) if focus_levels else 0

# Tính thời gian ở các mức tập trung
low_focus_time = sum(1 for f in focus_levels if f < 50) * (total_time / len(focus_levels))
medium_focus_time = sum(1 for f in focus_levels if 50 <= f < 80) * (total_time / len(focus_levels))
high_focus_time = sum(1 for f in focus_levels if f >= 80) * (total_time / len(focus_levels))

# Tính phần trăm thời gian
low_percent = (low_focus_time / total_time) * 100
medium_percent = (medium_focus_time / total_time) * 100
high_percent = (high_focus_time / total_time) * 100

# Đánh giá hiệu suất
if avg_focus >= 70 and understanding_score >= 70:
    performance_rating = "EXCELLENT"
    feedback = "Bạn đã duy trì sự tập trung và hiểu bài xuất sắc!"
elif avg_focus >= 60 and understanding_score >= 60:
    performance_rating = "GOOD"
    feedback = "Kết quả tốt, hãy tiếp tục phát huy!"
elif avg_focus >= 50 and understanding_score >= 50:
    performance_rating = "AVERAGE"
    feedback = "Kết quả trung bình, cần cải thiện sự tập trung"
else:
    performance_rating = "NEEDS IMPROVEMENT"
    feedback = "Hãy xem lại phương pháp học tập để cải thiện kết quả"

# --- Xuất báo cáo tổng kết ---
print("\n" + "=" * 50)
print("            BUỔI HỌC KẾT THÚC")
print("=" * 50)
print(f"Thời lượng học tập: {int(total_time // 60)} phút {int(total_time % 60)} giây")
print(f"Điểm tập trung trung bình: {avg_focus:.1f}%")
print(f"Điểm hiểu bài cuối cùng: {understanding_score}%")
print("\nPHÂN BỐ MỨC ĐỘ TẬP TRUNG:")
print(f"- Thấp (<50%): {low_percent:.1f}% thời gian")
print(f"- Trung bình (50-80%): {medium_percent:.1f}% thời gian")
print(f"- Cao (>80%): {high_percent:.1f}% thời gian")
print("\nĐÁNH GIÁ HIỆU SUẤT:")
print(f"- Xếp loại: {performance_rating}")
print(f"- Nhận xét: {feedback}")
print("=" * 50 + "\n")

# --- Vẽ biểu đồ kết hợp ---
plt.figure(figsize=(14, 10))

# Biểu đồ 1: Focus và Understanding
plt.subplot(2, 1, 1)
plt.plot(timestamps, focus_levels, label='Tập trung (%)', color='blue', alpha=0.7)
plt.plot(understanding_timestamps, understanding_scores, label='Hiểu bài (%)', color='green', marker='o', markersize=4)
plt.axhline(y=50, color='red', linestyle='--', label='Ngưỡng cảnh báo')
plt.title('DIỄN BIẾN TẬP TRUNG VÀ HIỂU BÀI')
plt.xlabel('Thời gian (s)')
plt.ylabel('Phần trăm (%)')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)

# Biểu đồ 2: Phân bố mức độ tập trung
plt.subplot(2, 1, 2)
focus_ranges = ['Thấp (<50%)', 'Trung bình (50-80%)', 'Cao (>80%)']
focus_percentages = [low_percent, medium_percent, high_percent]
colors = ['#ff9999', '#66b3ff', '#99ff99']

bars = plt.bar(focus_ranges, focus_percentages, color=colors)
plt.title('PHÂN BỐ THỜI GIAN THEO MỨC ĐỘ TẬP TRUNG')
plt.ylabel('Tỷ lệ thời gian (%)')

# Thêm giá trị phần trăm trên mỗi cột
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('learning_performance_summary.png')
plt.show()