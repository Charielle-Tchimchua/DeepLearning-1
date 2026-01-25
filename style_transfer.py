import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#  Charger les images 
content_image = load_and_preprocess_image("1.jpeg")  
style_image = load_and_preprocess_image("SPORT.jpeg")     

# 1. Charger VGG16
vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False

content_layers = [ ' block5_conv2 ']
style_layers = [ ' block1_conv1 ' , ' block2_conv1 ' , ' block3_conv1' , 'block4_conv1 ' , ' block5_conv1 ']

# 2. Définir les couches pour le style et le contenu
def create_extractor(vgg, style_layers, content_layers):
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    return keras.Model(inputs=vgg.input, outputs=outputs)

extractor = create_extractor ( vgg , style_layers , content_layers )

# 3. Charger et prétraiter les images
def load_and_preprocess_image(image_path, target_size=(512, 512)):
    img = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    return tf.convert_to_tensor(img)

# 4. Calculer les pertes de style et de contenu
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# 5. Définir les pertes de style et de contenu
def compute_loss(extractor, content_image, style_image, generated_image):
    content_outputs = extractor(content_image)
    style_outputs = extractor(style_image)
    generated_outputs = extractor(generated_image)
    content_loss = tf.reduce_mean(tf.square(content_outputs[-1] - generated_outputs[-1]))
    style_loss = 0
    for style_layer, content_layer in zip(style_outputs[:-1], generated_outputs[:-1]):
        style_gram = gram_matrix(style_layer)
        generated_gram = gram_matrix(content_layer)
        style_loss += tf.reduce_mean(tf.square(style_gram - generated_gram))
    total_loss = content_loss + 0.01 * style_loss  # Poids arbitraire pour le style loss
    return total_loss

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']
extractor = create_extractor(vgg, style_layers, content_layers)
print("VGG16 loaded. Ready for style transfer!")

