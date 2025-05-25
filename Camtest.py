import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
import matplotlib.pyplot as plt

# --- Load model cảm xúc của bạn ---
model = load_model('model_file.h5')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# --- Khởi tạo ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

last_focus_below_50 = 0
warning_threshold = 5  # giây

start_time = time.time()
tong_focus = 0
so_frame_co_mat = 0
focus_levels = []
timestamps = []
understanding_score = 50
last_understanding_update = 0
delay_between_understanding_update = 1.5  # giây


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


# Trả về điểm ảnh hưởng đến tập trung (score_mul) và mức thay đổi hiểu bài (understanding_delta)
def emotion_score(emotion_label):
    if emotion_label == "Unknown":
        return 0.0, -2
    elif emotion_label in ['Happy', 'Surprise']:
        return 1.0, 1
    elif emotion_label == 'Neutral':
        return 1.0, 0
    else:
        return 1.0, -1

def yaw_penalty(yaw):
    return 0.5 if abs(yaw) > 25 else 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 3)

    emotion_label = "Unknown"
    if len(faces) > 0:
        (x, y, w_face, h_face) = faces[0]
        sub_face_img = gray[y:y + h_face, x:x + w_face]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result_predict = model.predict(reshaped)
        label = np.argmax(result_predict, axis=1)[0]
        emotion_label = labels_dict[label]

    focus = 0
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        pitch, yaw, roll = estimate_pose(landmarks, w, h)

        score_mul, understanding_delta = emotion_score(emotion_label)
        focus = 100 * score_mul * yaw_penalty(yaw)
        focus = max(0, min(100, int(focus)))

    # Luôn tính focus trung bình
    tong_focus += focus
    so_frame_co_mat += 1
    focus_levels.append(focus)
    timestamps.append(time.time() - start_time)

    elapsed_time = time.time() - start_time
    avg_focus = tong_focus / so_frame_co_mat if so_frame_co_mat > 0 else 0

    current_time = time.time()

    # Cập nhật mức độ hiểu bài mỗi 1.5 giây
    if current_time - last_understanding_update > delay_between_understanding_update:
        # Nếu tập trung trung bình thấp hơn 50 thì trừ hiểu bài
        if avg_focus < 50:
            understanding_score -= 1  # Trừ dần, bạn có thể chỉnh số này
        else:
            # Cập nhật dựa theo cảm xúc (có thể là +1, 0, -1)
            _, understanding_delta = emotion_score(emotion_label)
            understanding_score += understanding_delta

        understanding_score = max(0, min(100, understanding_score))
        last_understanding_update = current_time

    if avg_focus < 50:
        cv2.putText(frame, "CANH BAO: Tap trung trung binh thap!", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 3)

    cv2.putText(frame, f"Cam xuc: {emotion_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Tap trung hien tai: {focus}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(frame, f"Tong thoi gian: {int(elapsed_time)}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
    cv2.putText(frame, f"Tap trung TB: {int(avg_focus)}%", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Muc do hieu: {understanding_score}%", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Phan tich tap trung", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Vẽ biểu đồ tập trung và hiểu bài theo thời gian ---
avg_focus_list = []
total = 0
for i, f in enumerate(focus_levels):
    total += f
    avg_focus_list.append(total / (i + 1))

plt.figure(figsize=(10, 5))
plt.plot(timestamps, avg_focus_list, label='Tập trung trung bình (%)', color='blue')
plt.axhline(y=50, color='red', linestyle='--', label='Ngưỡng cảnh báo')
plt.title('Biểu đồ Tập trung')
plt.xlabel('Thời gian (s)')
plt.ylabel('Tỉ lệ (%)')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
