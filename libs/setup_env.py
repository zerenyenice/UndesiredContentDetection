from dotenv import load_dotenv, find_dotenv
import os


class Env:
    directories = ['MODEL_DIR', 'FRAMES_DIR', 'VIDEO_DIR']

    @staticmethod
    def setup_environment_variables():
        load_dotenv(find_dotenv())

    @classmethod
    def create_directories(cls):
        for dir_i in cls.directories:
            if not os.path.exists(os.environ[dir_i]):
                os.makedirs(os.environ[dir_i], exist_ok=True)

    @classmethod
    def setup(cls):
        cls.setup_environment_variables()
        cls.create_directories()
