import tensorflow as tf
import os
import sys
import numpy as np
from PIL import Image


class TfRecorder:
    AUTO = tf.data.experimental.AUTOTUNE

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(self, img, label):
        feature = {
            'image_name': self._bytes_feature(img),
            'label': self._int64_feature(label),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def stack_two_sides(left,right):
        img_1, label_1 = left
        img_2, label_2 = right
        img = tf.concat([img_1,img_2],axis=0)
        label = tf.concat([label_1, label_2], axis=0)

        return img, label

    def create_tfrecord(self, path, save_path, label_dict):
        # todo change for json or csv label reader
        img_ls = os.listdir(path)
        # save_path = os.path.join(os.path.dirname(path), name)
        img_ls.sort()
        with tf.io.TFRecordWriter(save_path) as writer:
            for k, img_i in enumerate(img_ls):

                try:
                    label_csv = label_dict[img_i]
                except:
                    # print(f'passsing not found label --> {img_i}') todo unlock later
                    continue

                if label_csv == '0':
                    label = np.zeros((1, 1))
                elif label_csv == '1':
                    label = np.ones((1, 1))
                else:
                    raise ValueError('wrong file starting character')

                example = self.serialize_example(
                    os.path.join(path, img_i).encode(),
                    label)
                writer.write(example)
                #if k % 100 == 0: print(k, ', ', end='')

    def create_tfrecord_seperated_files(self, path, save_path, label_dict, label_type):
        # todo change for json or csv label reader
        img_ls = os.listdir(path)
        # save_path = os.path.join(os.path.dirname(path), name)
        img_ls.sort()
        with tf.io.TFRecordWriter(save_path) as writer:
            for k, img_i in enumerate(img_ls):

                try:
                    label_csv = label_dict[img_i]
                except:
                    # print(f'passsing not found label --> {img_i}') todo unlock later
                    continue

                if label_csv == '0':
                    label = np.zeros((1, 1))
                elif label_csv == '1':
                    label = np.ones((1, 1))
                else:
                    raise ValueError('wrong file starting character')

                if label_csv == label_type:
                    example = self.serialize_example(
                        os.path.join(path, img_i).encode(),
                        label)
                    writer.write(example)
                else:
                    continue
                #if k % 100 == 0: print(k, ', ', end='')

    @staticmethod
    def parse_tfrecord(example):
        features = {
            "image_name": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, features)
        img_raw = tf.io.read_file(example['image_name'])
        image = tf.io.decode_image(img_raw)
        image.set_shape([None, None, 3])

        image = tf.cast(image, tf.uint8)
        label = tf.cast(example['label'], tf.float32)
        label = tf.expand_dims(label, -1)
        return image, label

    def get_tf_record(self, tf_records):
        # Read from TFRecords. For optimal performance, we interleave reads from multiple files.
        return tf.data.TFRecordDataset(tf_records, num_parallel_reads=self.AUTO)

    def get_data(self, records):
        return records.map(self.parse_tfrecord, num_parallel_calls=self.AUTO)

    @staticmethod
    def get_data_size(data):
        return [i for i, _ in enumerate(data)][-1] + 1


if __name__ == '__main__':
    TfRecorder().create_tfrecord('data/frames', 'test.tfrec')
    print('a')
    filenames = ['data/test.tfrec']
    dat = TfRecorder().load_dataset(filenames)

    for i in dat.take(1):
        print(i)
