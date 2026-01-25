import tensorflow as tf 
from tensorflow import keras
import numpy as np

# 1. oad the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#   Number of classes
NUM_CLASSES = 10
INPUT_SHAPE = x_train.shape[1:]

# Normalize pixel values

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to One - Hot Encoding format
y_train = keras.utils.to_categorical( y_train , num_classes = NUM_CLASSES )
y_test = keras.utils.to_categorical( y_test , num_classes = NUM_CLASSES )
print(f" Input data shape : {INPUT_SHAPE} ")

print(f"Training labels shape after one-hot encoding: {y_train.shape}")
print(f"Test labels shape after one-hot encoding: {y_test.shape}")

# 2. Construction d'un CNN de base
def build_basic_cnn(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_basic_cnn ( INPUT_SHAPE , NUM_CLASSES )
model.compile( optimizer = 'adam'  , loss = ' categorica l_cr oss entr opy ', metrics =[ ' accuracy' ])

# Train the model
history = model . fit (
    x_train , y_train ,
    batch_size =64 ,
    epochs =10 ,
    validation_split =0.1 # 10% of training data for validation
)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# 3. Résidual Block pour ResNet
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    y = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
    if stride > 1:
        x = keras.layers.Conv2D(filters, (1, 1), strides=stride)(x)
    z = keras.layers.Add()([x, y])
    z = keras.layers.Activation('relu')(z)
    return z
model = build_basic_cnn ( INPUT_SHAPE , NUM_CLASSES )
model.compile( optimizer = 'adam'  , loss = ' categorical_cr oss entr opy ', metrics =[ ' accuracy' ])

# TODO: Build a small architecture using 3 consecutive residual blocks
def build_resnet(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = residual_block(inputs, 32)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)

# Ajout des couches finales
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=outputs)

resnet_model = build_resnet(INPUT_SHAPE, NUM_CLASSES)
resnet_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history_resnet = resnet_model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1
)