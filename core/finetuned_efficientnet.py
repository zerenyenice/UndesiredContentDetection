import tensorflow as tf
from PIL import Image
import numpy as np
import os


class UndesiredContentEfficientNet:
    """
        Class for loading model and running predictions.
    """
    undesired_content_model : tf.keras.Model = None
    model_name = 'efficient_net_b3_best.h5'
    weights_path = 'models'
    weights_path = os.path.join(weights_path, model_name)
    image_size = (296, 224)

    def __init__(self):
        # todo upload new model tf directory to bucket
        #self.undesired_content_model = tf.keras.models.load_model(UndesiredContentMobileNet.weights_path, custom_objects={'KerasLayer': hub.KerasLayer})
        self.undesired_content_model = tf.keras.models.load_model(self.weights_path)

    def _preprocess(self, image):
        """
            inputs:
                image: rgb image [H,W,C] image
        """

        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])
        image = np.expand_dims(image, 0)

        return image

    def predict(self, image):
        """
            inputs:
                image: bgr pre-processed 4 dims [N,H,W,C] image
        """

        input_image = self._preprocess(image)
        model_pred = self.undesired_content_model.predict(input_image)

        return model_pred

if __name__ == "__main__":
    nsf = UndesiredContentEfficientNet()
    image = Image.open('data/test.jpg')
    image = np.array(image).astype('uint8')
    nsf.predict(image)