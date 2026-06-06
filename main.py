import os
import keras

os.environ["KERAS_BACKEND"] = "tensorflow"

# Loading Data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Shape of x_train, y_train, x_test, y_test:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reshaping Data
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))
print("Shape after reshaping:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Creating model
model = keras.models.Sequential([
    keras.Input(shape=(28 * 28,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
)

# Training the model
batch_size = 64
epochs = 20

# Stop training early if validation loss doesn't improve for 3 consecutive epochs
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=callbacks
)

# Evaluating on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the trained model
model.save("mnist_model.keras")
print("Model saved as mnist_model.keras")
