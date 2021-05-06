import tensorflow as tf
from tensorflow import keras
import numpy as np
import imutils
import os


class Model:

    saved_model = 'model_25.h5'

    def __init__(self):
        self.model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model))
        self.model.summary()

    def preprocess(self, image):
        #image = tf.image.resize(image, [84, 112])
        image = tf.image.resize(image, [60, 80])
        return image

    def predict(self, image):
        image = self.preprocess(image)
        angle, speed = self.model.predict(np.array([image]))[0]
        # Training data was normalised so convert back to car units
        Angles = [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1]
        NewAngles  = Angles - angle*np.ones([1, len(Angles)])
        angle = Angles[np.argmin(np.absolute(NewAngles))]
        if speed > 0.5:
          speed = 1
        elif speed < 0.5:
          speed = 0 
        angle = 80 * angle + 50
        speed = 35 * speed
        return angle, speed
