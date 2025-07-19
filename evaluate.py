import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd

# Cấu hình đường dẫn và tham số
test_path = 'data/test'
IMG_SIZE_CUSTOM = (48, 48)  # Cho CustomCNN
IMG_SIZE_LARGE = (224, 224)  # Cho MobileNetV2 và ResNet50
BATCH_SIZE = 32


# Định nghĩa metrics phù hợp với phiên bản TF
def get_metrics(num_classes):
    return [
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),  # Bỏ tham số average
        tf.keras.metrics.Recall(name='recall'),  # Bỏ tham số average
        tf.keras.metrics.AUC(name='auc')  # Bỏ multi_label nếu cần
    ]


# Tạo data generators
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generator cho CustomCNN (48x48)
test_custom = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE_CUSTOM,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Generator cho MobileNetV2 và ResNet50 (224x224)
test_large = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE_LARGE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Đường dẫn đến các file .h5 (thay bằng đường dẫn thực tế)
model_paths = {
    'CustomCNN': 'customcnn.h5',
    'MobileNetV2': 'mobilenetv2.h5',
    'ResNet50V2': 'resnet50v.h5'
}

# Đánh giá và thu thập kết quả
results = []
for name, path in model_paths.items():
    try:
        # Load mô hình
        model = load_model(path)

        # Chọn dataset phù hợp
        test_data = test_custom if name == 'CustomCNN' else test_large

        # Biên dịch lại với metrics
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=get_metrics(len(test_data.class_indices)))

        # Đánh giá mô hình
        eval_results = model.evaluate(test_data, verbose=0)

        # Lưu kết quả
        results.append({
            'Model': name,
            'Loss': eval_results[0],
            'Accuracy': eval_results[1],
            'Precision': eval_results[2],
            'Recall': eval_results[3],
            'AUC': eval_results[4]
        })

        print(f"Đã đánh giá xong {name}")
    except Exception as e:
        print(f"Lỗi khi đánh giá {name}: {str(e)}")

# Tạo DataFrame từ kết quả
results_df = pd.DataFrame(results)
print("\nKết quả so sánh:")
print(results_df.to_markdown(floatfmt=".4f"))


# Vẽ biểu đồ so sánh
def plot_comparison(df, save_path=None):
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH', fontsize=16)

    colors = ['#FF9AA2', '#FFB7B2', '#FFDAC1']  # Màu pastel

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        bars = ax.bar(df['Model'], df[metric], color=colors)
        ax.set_title(metric)
        ax.set_ylim(0, 1)

        # Thêm giá trị trên mỗi cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Xuất biểu đồ
plot_comparison(results_df, save_path='model_comparison.png')