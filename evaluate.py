import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Config ---
# Kiểm tra input_shape của từng model trước khi set giá trị
IMG_GRAY = (48, 48)  # Cho Custom CNN (grayscale)
IMG_RGB_MOBILE = (224, 224)  # Cho MobileNetV2 (đã xác nhận model được train với 128x128)
IMG_RGB_RESNET = (224, 224)  # Cho ResNet50 (chuẩn ImageNet)
BATCH_SIZE = 32
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
test_path = 'data/test'  # Thay đổi thành đường dẫn thực tế


# --- Khởi tạo generators ---
def create_generators():
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    generators = {
        'custom_cnn': test_datagen.flow_from_directory(
            test_path,
            target_size=IMG_GRAY,
            color_mode='grayscale',
            classes=CLASSES,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        ),
        'mobilenet': test_datagen.flow_from_directory(
            test_path,
            target_size=IMG_RGB_MOBILE,
            color_mode='rgb',
            classes=CLASSES,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        ),
        'resnet': test_datagen.flow_from_directory(
            test_path,
            target_size=IMG_RGB_RESNET,
            color_mode='rgb',
            classes=CLASSES,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    }
    return generators


# --- Load và kiểm tra models ---
def load_models():
    models = {}

    try:
        models['custom_cnn'] = tf.keras.models.load_model('customcnn.h5')
        print("Custom CNN loaded. Input shape:", models['custom_cnn'].input_shape)
    except Exception as e:
        print(f"Lỗi khi load Custom CNN: {str(e)}")

    try:
        models['mobilenet'] = tf.keras.models.load_model('mobilenetv2.h5')
        print("MobileNetV2 loaded. Input shape:", models['mobilenet'].input_shape)
    except Exception as e:
        print(f"Lỗi khi load MobileNetV2: {str(e)}")

    try:
        models['resnet'] = tf.keras.models.load_model('resnet50.h5')
        print("ResNet50 loaded. Input shape:", models['resnet'].input_shape)
    except Exception as e:
        print(f"Lỗi khi load ResNet50: {str(e)}")

    return models


# --- Đánh giá model ---
def evaluate_model(model, generator):
    if model is None or generator is None:
        return None, None, None

    # Đảm bảo generator được reset trước khi đánh giá
    generator.reset()

    # Đánh giá
    loss, acc = model.evaluate(generator, verbose=0)
    y_pred = np.argmax(model.predict(generator), axis=1)
    return loss, acc, y_pred


# --- Visualization ---
def plot_results(model_names, accuracies, losses):
    # Biểu đồ accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'orange', 'green'])
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')

    for bar, acc in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{acc:.2f}%", ha='center')

    # Biểu đồ loss
    plt.subplot(1, 2, 2)
    plt.bar(model_names, losses, color=['skyblue', 'orange', 'green'])
    plt.ylabel('Loss')
    plt.title('Model Loss Comparison')

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower()}.png')
    plt.show()


# --- Main execution ---
def main():
    # Khởi tạo
    generators = create_generators()
    models = load_models()

    # Lấy nhãn thực
    y_true = generators['resnet'].classes

    # Đánh giá từng model
    results = []
    predictions = {}

    print("\nĐang đánh giá các mô hình...")

    # Custom CNN
    if 'custom_cnn' in models:
        loss, acc, y_pred = evaluate_model(models['custom_cnn'], generators['custom_cnn'])
        if loss is not None:
            results.append(('Custom CNN', acc * 100, loss))
            predictions['custom_cnn'] = y_pred
            print(f"Custom CNN - Accuracy: {acc * 100:.2f}%")

    # MobileNetV2
    if 'mobilenet' in models:
        loss, acc, y_pred = evaluate_model(models['mobilenet'], generators['mobilenet'])
        if loss is not None:
            results.append(('MobileNetV2', acc * 100, loss))
            predictions['mobilenet'] = y_pred
            print(f"MobileNetV2 - Accuracy: {acc * 100:.2f}%")

    # ResNet50
    if 'resnet' in models:
        loss, acc, y_pred = evaluate_model(models['resnet'], generators['resnet'])
        if loss is not None:
            results.append(('ResNet50', acc * 100, loss))
            predictions['resnet'] = y_pred
            print(f"ResNet50 - Accuracy: {acc * 100:.2f}%")

    # Lưu kết quả
    if results:
        df = pd.DataFrame(results, columns=['Model', 'Accuracy (%)', 'Loss'])
        df.to_excel('fer_results.xlsx', index=False)

        # Visualize
        model_names = [x[0] for x in results]
        accuracies = [x[1] for x in results]
        losses = [x[2] for x in results]

        plot_results(model_names, accuracies, losses)

        # Confusion matrix cho model tốt nhất
        best_model = max(results, key=lambda x: x[1])[0].lower()
        if best_model in predictions:
            plot_confusion_matrix(y_true, predictions[best_model], best_model)
    else:
        print("Không có model nào được đánh giá thành công!")


if __name__ == "__main__":
    main()