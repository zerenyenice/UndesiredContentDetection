import tensorflow as tf
import tensorflow_addons as tfa
import imgaug.augmenters as iaa


class Augmentor:

    x_y_shift = (-0.20, 0.20)
    rotation_angle = (-25,25)
    horizontal_flip = 0.5
    shear_degree = (-20,20)
    zoom_perc = (0.80,1.20)

    def __init__(self):

        self.aug_pipe = None

    def init_augmentation_pipeline(self):
        self.aug_pipe = iaa.Sequential(
            [
                iaa.Sometimes(0.30,
                              iaa.OneOf([
                                  iaa.imgcorruptlike.GaussianNoise((1, 3)),
                                  iaa.imgcorruptlike.ShotNoise((1, 3)),
                                  iaa.imgcorruptlike.ImpulseNoise((1, 3)),
                                  iaa.imgcorruptlike.SpeckleNoise((1, 3)),
                                  iaa.imgcorruptlike.GaussianBlur(1),
                                  iaa.imgcorruptlike.MotionBlur((1, 2)),
                                  iaa.imgcorruptlike.Brightness((1, 3)),
                                  iaa.imgcorruptlike.Pixelate((1, 2))]),
                              iaa.Sometimes(0.40,
                                            iaa.imgcorruptlike.JpegCompression((1, 3))
                                            )),
                iaa.Fliplr(self.horizontal_flip),
                iaa.Affine(scale=self.zoom_perc,
                           translate_percent={'x':self.x_y_shift,'y':self.x_y_shift},
                           rotate=self.rotation_angle,
                           order=3,
                           mode='constant',
                           backend='cv2'),
                iaa.Rot90((1,4),keep_size=False)
            ]
        )

    def single_image_augmentation(self, image):
        return self.aug_pipe(image=image)



if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    img = np.array(Image.open('data/frames/ns00001.jpg'))
    img = tf.image.rot90(img,1).numpy()
    i_sized = tf.image.resize_with_pad(
        img, 336,224
    ).numpy()
    a = Augmentor()
    a.augmentation_pipeline()

    d = a.aug_pipe(image=img)
    out = Image.fromarray(d)
    out.show()

    out = Image.fromarray(i_sized.astype('uint8'))
    out.show()
