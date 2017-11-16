from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from keras.preprocessing import image
from keras.applications.resnet50 import *

model = ResNet50(weights=None)
model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5py')

img = image.load_img('data/deer.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=5)[0])
