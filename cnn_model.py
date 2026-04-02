import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np

# ---------------- CONFIG ----------------
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS_FROZEN   = 15
EPOCHS_FINETUNE = 25
DATASET_PATH = "../dataset"
MODEL_PATH   = "../models/cnn_model.h5"

os.makedirs("../models", exist_ok=True)

# ----------------------------------------------------------------
# IMPORTANT: EfficientNetB0 has its own internal normalization.
# Do NOT use rescale=1/255 — pass raw [0,255] pixels.
# Using rescale causes double-normalization → model sees wrong range.
# ----------------------------------------------------------------

train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.08,
    fill_mode="reflect"
)

val_datagen = ImageDataGenerator(
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print(f"Classes : {train_gen.class_indices}")
print(f"Train   : {train_gen.samples} | Val: {val_gen.samples}")

# ----------------------------------------------------------------
# CLASS WEIGHTS
# ----------------------------------------------------------------
counts = np.array([
    len(os.listdir(os.path.join(DATASET_PATH, c)))
    for c in ["healthy", "inflamed", "infected"]
])
total = counts.sum()
class_weights = {i: total / (3.0 * cnt) for i, cnt in enumerate(counts)}
print(f"Class counts  : {dict(zip(['healthy','inflamed','infected'], counts))}")
print(f"Class weights : {class_weights}")

# ---------------- MODEL ----------------
def build_model():
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    # ImageDataGenerator produces floats in [0,1] (no rescale set).
    # EfficientNet expects [0,255], so scale back up.
    x = tf.keras.layers.Rescaling(scale=255.0)(inputs)
    x = base(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

# ---------------- PHASE 1: FROZEN ----------------
print("\n=== Phase 1: Frozen base ===")
model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cb1 = [
    EarlyStopping(patience=6, restore_best_weights=True,
                  monitor="val_accuracy", verbose=1),
    ReduceLROnPlateau(factor=0.4, patience=3, min_lr=1e-6, verbose=1),
]

h1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weights,
    callbacks=cb1
)

print(f"\nPhase 1 best val acc: {max(h1.history['val_accuracy']):.4f}")

# ---------------- PHASE 2: FINE-TUNE ----------------
print("\n=== Phase 2: Unfreeze last 30 layers ===")

# Layer index 2 = EfficientNetB0 (0=Input, 1=Rescaling, 2=efficientnetb0)
base_model = model.layers[2]
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep EfficientNet BatchNorm layers frozen (small dataset — don't shift running stats)
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print(f"Trainable weight tensors: {len(model.trainable_variables)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cb2 = [
    EarlyStopping(patience=8, restore_best_weights=True,
                  monitor="val_accuracy", verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True,
                    monitor="val_accuracy", verbose=1)
]

h2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weights,
    callbacks=cb2
)

best = max(max(h1.history["val_accuracy"]), max(h2.history["val_accuracy"]))
print(f"\n✅ Saved → {MODEL_PATH}")
print(f"Phase 1 best: {max(h1.history['val_accuracy']):.4f}")
print(f"Phase 2 best: {max(h2.history['val_accuracy']):.4f}")
print(f"Overall best: {best:.4f}")