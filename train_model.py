import os
import mlflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # désactive toute tentative GPU
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import numpy as np

optimizers = {
    'SGD_with_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': keras.optimizers.RMSprop(),
    'Adam': keras.optimizers.Adam()
}
# Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Diviser en training et validation
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]

# Normalisation des données
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_val = x_val.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

for opt_name, optimizer in optimizers.items():
    with mlflow.start_run(run_name=f"Optimizer_Comparison_{opt_name}"):
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dense(10, activation='softmax')
        ])

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))

# Évaluation du modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
mlflow.log_param("optimizer", opt_name)
mlflow.log_metric("final_test_accuracy", test_acc)
print(f"Précision sur les données de test: {test_acc:.4f}")

# Sauvegarde du modèle
model.save("mnist_model.h5")
print("Modèle sauvegardé sous mnist_model.h5")
