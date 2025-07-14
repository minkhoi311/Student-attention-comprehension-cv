# --- File 1: train_custom_cnn.py ---
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, regularizers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Config ---
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
train_path = 'data/train'
test_path = 'data/test'
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Data ---
# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Thêm rescale vào train luôn
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_path, target_size=IMG_SIZE, color_mode='grayscale',
                                              classes=CLASSES, batch_size=BATCH_SIZE, class_mode='categorical')

val_gen = val_datagen.flow_from_directory(test_path, target_size=IMG_SIZE, color_mode='grayscale',
                                          classes=CLASSES, batch_size=BATCH_SIZE, class_mode='categorical')

# --- Class Weights ---
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_gen.classes),
                                                  y=train_gen.classes)
class_weights_dict = dict(enumerate(class_weights))

# --- Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.2,  # Factor by which the learning rate will be reduced
    patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-5  # Minimum learning rate
)

# --- Model ---
def build_custom_cnn():
    #tang 1
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'))
    model.add(layers.BatchNormalization())  # Add batch normalization
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))  # Add dropout
    #tang 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    #tang 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    #tang 4
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
#    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Khởi tạo model từ hàm build_custom_cnn
model = build_custom_cnn()
model.summary()

# --- Training ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr]
)

# --- Save Model ---
model.save('custom_cnn_fer.h5')

# ---- ĐÁNH GIÁ MÔ HÌNH ----
print("\n=== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION ===")

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs('result_ccnn', exist_ok=True)

# Tải lại mô hình tốt nhất (nếu có)
if os.path.exists('result_ccnn/best_model.h5'):
    best_model = tf.keras.models.load_model('result_ccnn/best_model.h5')
    print("Sử dụng mô hình tốt nhất từ checkpoint: result_ccnn/best_model.h5")
else:
    best_model = model
    print("Không tìm thấy mô hình tốt nhất → dùng mô hình cuối cùng vừa huấn luyện")

# Reset generator và dự đoán
val_gen.reset()
y_pred = best_model.predict(val_gen, verbose=1)
y_true = val_gen.classes
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
report = classification_report(y_true, y_pred_classes, target_names=CLASSES, digits=4)
print("Classification Report:\n", report)

# Lưu report ra file
with open('result_ccnn/classification_report.txt', 'w') as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            annot_kws={"size": 10})

plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('result_ccnn/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Confusion matrix và báo cáo phân loại đã được lưu trong thư mục 'result_ccnn/'.")
