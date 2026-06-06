import os
import numpy as np
import keras
from keras import layers

os.environ["KERAS_BACKEND"] = "tensorflow"

# ── Load Data ─────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Original shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# ── Reshape & Normalize ───────────────────────────────────────────────────
# CNN expects (batch, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
print("Reshaped:", x_train.shape, x_test.shape)

# ── Data Augmentation ─────────────────────────────────────────────────────
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),           # rotate up to 10%
    layers.RandomZoom(0.1),               # zoom up to 10%
    layers.RandomTranslation(0.1, 0.1),   # shift up to 10%
], name="data_augmentation")

# ── CNN Model ─────────────────────────────────────────────────────────────
inputs = keras.Input(shape=(28, 28, 1))

x = data_augmentation(inputs)

# Block 1
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

# Block 2
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

# Block 3 — deeper layers
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

# Fully connected head
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)
print(model.summary())

# ── Compile ───────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
)

# ── Callbacks ─────────────────────────────────────────────────────────────
callbacks = [
    # Stop if val_loss doesn't improve for 5 epochs
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # Reduce learning rate if val_loss plateaus
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# ── Train ─────────────────────────────────────────────────────────────────
batch_size = 64
epochs = 30

model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=callbacks
)

# ── Evaluate ──────────────────────────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ── Save ──────────────────────────────────────────────────────────────────
model.save("mnist_model.keras")
print("Model saved as mnist_model.keras")
