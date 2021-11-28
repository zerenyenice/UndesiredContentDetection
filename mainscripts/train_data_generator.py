import os
from libs.video_editor import extract_frames
from glob import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataGenerator:
    """
    Working with mp4 and jpg formats

    Generate frame data from videos for labeling process.
    put generated data under a
    """

    def __init__(self,
                 output_path,
                 nth_frame=1,
                 data_ratios=(0.70, 0.15, 0.15)):

        self.temp_frame_path = os.environ['FRAMES_DIR']
        self.extraction_path = os.path.join(output_path, 'dataset')
        self.training_folder_path = os.path.join(output_path, 'train_set')
        self.validation_folder_path = os.path.join(output_path, 'validate_set')
        self.test_folder_path = os.path.join(output_path, 'test_set')
        self.output_path = output_path
        self.nth_frame = nth_frame
        self.train_ratio, self.validate_ratio, self.test_ratio = data_ratios
        self._create_output_dir()

        # todo change tfrecord creator for json or csv label file

    def _extract_name_and_move_frames(self, video):
        v_i_base, _ = self._get_base_name(video)
        extract_frames(video, frame_n_size=self.nth_frame)
        for img_i in glob(os.path.join(self.temp_frame_path, '*.jpg')):
            img_base_i, img_suffix_i = self._get_base_name(img_i)
            img_base_i = '%05d' % (int(img_base_i) * self.nth_frame)
            shutil.move(img_i, os.path.join(self.extraction_path, f'{v_i_base}_{img_base_i}.{img_suffix_i}'))

    def run_pre_labeling_data_generator(self, video_path):
        self._create_sub_dir(self.extraction_path)

        video_list = glob(os.path.join(video_path, '*.mp4'))


        for v_i in tqdm(video_list, total=len(video_list), desc="video_loop", position=0, leave=False):
            self._extract_name_and_move_frames(v_i)

    def run_data_splitter(self, data_path=None):

        if data_path is None:
            data_path=self.extraction_path

        self._create_sub_dir(self.training_folder_path)
        self._create_sub_dir(self.validation_folder_path)
        self._create_sub_dir(self.test_folder_path)

        image_list = glob(os.path.join(data_path, '*.jpg'))
        base_names = [os.path.basename(x).split('_')[0] for x in image_list]
        base_names_unique = np.unique(base_names)

        train_set, test_set = train_test_split(base_names_unique,
                                               train_size=self.train_ratio,
                                               test_size=self.test_ratio+self.validate_ratio,
                                               random_state=1,
                                               shuffle=True
                                               )

        test_set, validate_set = train_test_split(test_set,
                                                  train_size=self.test_ratio/(self.test_ratio+self.validate_ratio),
                                                  test_size=self.validate_ratio/(self.test_ratio+self.validate_ratio),
                                                  random_state=1,
                                                  shuffle=True)


        print(len(train_set), len(validate_set), len(test_set))

        for base_i, image_i in zip(base_names, image_list):
            if base_i in train_set:
                shutil.copy(image_i, os.path.join(self.training_folder_path, os.path.basename(image_i)))
            elif base_i in validate_set:
                shutil.copy(image_i, os.path.join(self.validation_folder_path, os.path.basename(image_i)))
            elif base_i in test_set:
                shutil.copy(image_i, os.path.join(self.test_folder_path, os.path.basename(image_i)))

            else:
                raise RuntimeError('check for bug, all base names should be splited')

    @staticmethod
    def _get_base_name(path):
        return os.path.basename(path).split('.')

    def _create_output_dir(self):

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    @staticmethod
    def _create_sub_dir(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)


if __name__ == '__main__':
    from libs.setup_env import Env
    Env.setup_environment_variables()
    a = DataGenerator(output_path='data/test2')
    a._data_spliter('data/train_frames/validation')