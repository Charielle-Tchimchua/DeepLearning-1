import tensorflow as tf
from tensorflow import keras

def conv_block(input_tensor, num_filter):
    x = keras.layers.Conv2D(num_filter, (3, 3), padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(128, 128, 1)):
    inputs = keras.Input(input_shape)

    # Encoder Path
    c1 = conv_block(inputs, 32)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    # Bridge
    b = conv_block(p3, 256)

    # Decoder Path
    u1 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = keras.layers.Concatenate()([u1, c3])
    d1 = conv_block(u1, 128)

    u2 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = keras.layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)

    u3 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = keras.layers.Concatenate()([u3, c1])
    d3 = conv_block(u3, 32)

    # Output Layer
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)

    return keras.Model(inputs=[inputs], outputs=[outputs])

# Compilation
model = build_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_coeff, iou_metric])
