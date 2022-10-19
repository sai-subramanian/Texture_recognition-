import random
from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow import *
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import os
from tensorflow.keras import optimizers
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(tensor)

content_img = Image.open("download (1).jpg")
generated_image = Variable(convert_to_tensor(content_img, dtype=float32))
noise = random.uniform(shape(generated_image), -0.25, 0.25)
generated_image = add(noise, generated_image)
generated_image = tensor_to_image(generated_image)

plt.imshow(generated_image)
plt.show()