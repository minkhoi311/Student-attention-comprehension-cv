from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# --- Config ---
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30
train_path = 'data/train'
test_path = 'data/test'
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Data ---
# Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2,
                                    rotation_range=15,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

#data generator
train_dataset  = train_datagen.flow_from_directory(directory = train_path,
                                                   target_size = IMG_SIZE,
                                                   color_mode='grayscale',
                                                   class_mode = 'categorical',
                                                   subset = 'training',
                                                   batch_size = BATCH_SIZE,
                                                   shuffle = True)

valid_dataset = valid_datagen.flow_from_directory(directory = train_path,
                                                  target_size = IMG_SIZE,
                                                  color_mode='grayscale',
                                                  class_mode = 'categorical',
                                                  subset = 'validation',
                                                  batch_size = BATCH_SIZE,
                                                  shuffle = False)
# --- Model ---
INPUT_SHAPE = (48, 48, 1)
model = keras.models.Sequential([
    keras.Input(shape=INPUT_SHAPE),
    # 32
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', ),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    # 64
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', ),
    keras.layers.LayerNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', ),
    keras.layers.LayerNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    # 128
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', ),
    keras.layers.LayerNormalization(),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', ),
    keras.layers.LayerNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    # 256
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', ),
    keras.layers.LayerNormalization(),
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', ),
    keras.layers.LayerNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    # 512
    keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', ),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', ),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    # 1024
    keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same', ),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same', ),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),

    # Dense layers
    # Global Average Pooling layer is used instead of Flatten as it is more effiecient
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(CLASSES), activation='softmax')
])

#Helper function
def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
        f1_score,
]
#Call back
lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 20,verbose = 1,factor = 0.50, min_lr = 1e-10)
mcp = ModelCheckpoint('customecnn_mymodel.h5')
es = EarlyStopping(verbose=1, patience=20)
#save
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=METRICS)
#
model.summary()

#run
history=model.fit(train_dataset,validation_data=valid_dataset,epochs = EPOCHS,verbose = 1,callbacks=[lrd,mcp,es])
#save
model.save('custom_cnn.h5')

# ---- ĐÁNH GIÁ MÔ HÌNH ----
def Train_Val_Plot(acc, val_acc, loss, val_loss, auc, val_auc, precision, val_precision, f1, val_f1, save_path=None):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])

    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])

    ax3.plot(range(1, len(auc) + 1), auc)
    ax3.plot(range(1, len(val_auc) + 1), val_auc)
    ax3.set_title('History of AUC')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('AUC')
    ax3.legend(['training', 'validation'])

    ax4.plot(range(1, len(precision) + 1), precision)
    ax4.plot(range(1, len(val_precision) + 1), val_precision)
    ax4.set_title('History of Precision')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision')
    ax4.legend(['training', 'validation'])

    ax5.plot(range(1, len(f1) + 1), f1)
    ax5.plot(range(1, len(val_f1) + 1), val_f1)
    ax5.set_title('History of F1-score')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('F1 score')
    ax5.legend(['training', 'validation'])

    plt.savefig(save_path)
    plt.show()


Train_Val_Plot(history.history['accuracy'], history.history['val_accuracy'],
               history.history['loss'], history.history['val_loss'],
               history.history['auc'], history.history['val_auc'],
               history.history['precision'], history.history['val_precision'],
               history.history['f1_score'], history.history['val_f1_score'],
               save_path='metrics_plot.png'
               )