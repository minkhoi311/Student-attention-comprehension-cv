from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# ---- CẤU HÌNH ----
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

val_datagen = ImageDataGenerator(rescale=1. / 255)

# ---- TẠO DATA GENERATORS ----
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    classes=CLASSES,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    classes=CLASSES,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

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

# ---- RESNET18 BLOCK khối cơ bản trong resnet ----
def conv_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# ---- BUILD RESNET18 ----
def build_resnet18(input_shape=(48, 48, 1), num_classes=7):
    inputs = layers.Input(shape=input_shape) #layer1
    x = layers.Conv2D(32, kernel_size=5, strides=2, padding='same', kernel_initializer='he_normal')(inputs)  # nhẹ hơn 7x7
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # ResNet Block: 32, 64, 128, 256 filters,khoi residual
    for filters, stride in zip([32, 64, 128, 256], [1, 2, 2, 2]):
        x = conv_block(x, filters, stride=stride)
        x = conv_block(x, filters, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Khởi tạo model ---
model = build_resnet18(input_shape=(48, 48, 1), num_classes=7)
model.summary()

# --- Training ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr]
)

# --- Save model ---
model.save('resnet18_fer.h5')

# ---- ĐÁNH GIÁ MÔ HÌNH ----
print("\n=== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION ===")

# Tạo thư mục lưu kết quả
os.makedirs('result_resnet18', exist_ok=True)

best_model = model
print("Dùng mô hình cuối cùng vừa huấn luyện")

# Reset generator và dự đoán
val_gen.reset()
y_pred = best_model.predict(val_gen, verbose=1)
y_true = val_gen.classes
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
report = classification_report(y_true, y_pred_classes, target_names=CLASSES, digits=4)
print("Classification Report:\n", report)

# Lưu report ra file
with open('result_resnetv18/classification_report.txt', 'w') as f:
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
plt.savefig('result_resnetv18/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ---- VẼ TRAINING HISTORY ----
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Vẽ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Vẽ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('result_resnetv18/training_history.png', dpi=300)
    plt.show()

plot_training_history(history)

print("Confusion matrix và báo cáo phân loại đã được lưu trong thư mục 'result_resnetv18/'.")