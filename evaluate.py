import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Config ---
IMG_SIZE_GRAY = (48, 48)      # Cho Custom CNN
IMG_SIZE_RGB = (128, 128)     # Cho MobileNetV2
BATCH_SIZE = 32
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
test_path = 'data/test'

# --- Load test data ---
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Custom CNN: ảnh grayscale 48x48
test_gen_gray = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE_GRAY,
    color_mode='grayscale',
    classes=CLASSES,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# MobileNetV2: ảnh RGB 128x128
test_gen_rgb = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE_RGB,
    color_mode='rgb',
    classes=CLASSES,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Load models ---
model_custom = tf.keras.models.load_model('model_file.h5')
model_mobile = tf.keras.models.load_model('mobilenetv2_fer.h5')

# --- Evaluate ---
print("\nĐang đánh giá mô hình Custom CNN...")
loss1, acc1 = model_custom.evaluate(test_gen_gray, verbose=0)

print("Đang đánh giá mô hình MobileNetV2...")
loss2, acc2 = model_mobile.evaluate(test_gen_rgb, verbose=0)

# --- Print results ---
print("\n=== KẾT QUẢ ===")
print(f"Custom CNN Accuracy:    {acc1 * 100:.2f}%")
print(f"MobileNetV2 Accuracy:   {acc2 * 100:.2f}%")

# --- Plot ---
model_names = ['Custom CNN', 'MobileNetV2']
accuracies = [acc1 * 100, acc2 * 100]

plt.figure(figsize=(7, 5))
bars = plt.bar(model_names, accuracies, color=['skyblue', 'orange'])
plt.ylim(0, 100)
plt.ylabel('Test Accuracy (%)')
plt.title('Model Accuracy Comparison on FER2013')

# Ghi giá trị trên đầu cột
for bar, acc in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{acc:.2f}%", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('fer_model_accuracy_comparison_2models.png')
plt.show()
