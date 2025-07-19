from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import keras
import matplotlib.pyplot as plt

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

# # --- Class Weights ---
# class_weights = class_weight.compute_class_weight(
#     'balanced',
#     classes=np.unique(train_gen.classes),
#     y=train_gen.classes)
#
# class_weights_dict = dict(enumerate(class_weights))

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
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
]
#Call back
lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 20,verbose = 1,factor = 0.50, min_lr = 1e-10)
mcp = ModelCheckpoint('customecnn_mymodel.h5')
es = EarlyStopping(verbose=1, patience=20)
#save
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=METRICS)
#
model.summary()
#Hien anh
# from tensorflow.keras.utils import plot_model
# from IPython.display import Image
# plot_model(model, to_file='mymodel2.png', show_shapes=True,show_layer_names=True)
# Image(filename='mymodel2.png')

#run
history=model.fit(train_dataset,validation_data=valid_dataset,epochs = EPOCHS,verbose = 1,callbacks=[lrd,mcp,es])
#save
model.save('custom_cnn.h5')

# ---- ĐÁNH GIÁ MÔ HÌNH ----
## plotting Results

def Train_Val_Plot(acc, val_acc, loss, val_loss, auc, val_auc, precision, val_precision, save_path=None):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("MODEL'S METRICS VISUALIZATION")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['Training', 'Validation'])

    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['Training', 'Validation'])

    ax3.plot(range(1, len(auc) + 1), auc)
    ax3.plot(range(1, len(val_auc) + 1), val_auc)
    ax3.set_title('AUC')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('AUC')
    ax3.legend(['Training', 'Validation'])

    ax4.plot(range(1, len(precision) + 1), precision)
    ax4.plot(range(1, len(val_precision) + 1), val_precision)
    ax4.set_title('Precision')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision')
    ax4.legend(['Training', 'Validation'])

    if save_path:
        plt.savefig(save_path)
    plt.show()

Train_Val_Plot(
    history.history['accuracy'], history.history['val_accuracy'],
    history.history['loss'], history.history['val_loss'],
    history.history['auc'], history.history['val_auc'],
    history.history['precision'], history.history['val_precision'],
    save_path='metrics_plot.png'
)