from libs.setup_env import Env
import argparse
import logging
import os


class MainScript:
    Env.setup_environment_variables()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=os.environ['LOG_LEVEL'])

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    @staticmethod
    def _process_data_generate(arguments):
        from mainscripts.train_data_generator import DataGenerator
        dg = DataGenerator(
            arguments.output_path,
            arguments.nth_frame,
            arguments.split_ratios
        )
        if arguments.process == 'frame_generate':
            dg.run_pre_labeling_data_generator(arguments.input_path)
        elif arguments.process == 'data_split':
            dg.run_data_splitter(arguments.input_path)
        elif arguments.process == 'both':
            dg.run_pre_labeling_data_generator(arguments.input_path)
            dg.run_data_splitter()
        else:
            raise AttributeError('wrong process selection')

    @classmethod
    def _parser_data_generate(cls):

        cls.p = cls.subparsers.add_parser("data_generate", help="Generate training data for fine-tuning undesired_content models.")

        cls.p.add_argument('--process', '-p', required=True, dest="process",
                           choices=['frame_generate', 'data_split', 'both'],
                           help="process selection for data generation")
        cls.p.add_argument('--input-path', '-ip', required=True, dest="input_path",
                           help="An input directory containing the mp4 files or frames for processing. "
                                "if running for 'both' option, give video path as input_path")
        cls.p.add_argument('--output-path', '-op', required=True, dest="output_path",
                           help="Output frame directory where the extracted jpg images will be stored.")
        cls.p.add_argument('--nth-frame', '-nf', required=False, default=1, type=int, dest="nth_frame",
                           help="Option for extracting and saving every nth frame.")
        cls.p.add_argument('--split-ratios', '-sr', required=False, default=(0.70, 0.15, 0.15), type=tuple,
                           dest="split_ratios",
                           help="split ratios for training,validation and test sets")

        cls.p.set_defaults(func=cls._process_data_generate)

    @staticmethod
    def _process_training(arguments):

        if arguments.backbone == 'mobilenet':
            from core.undesired_content_mobilenet import UndesiredContentMobileNet
            from mainscripts.fine_tuning.mobilenet import FTMobilenet

            model = UndesiredContentMobileNet()
            model = FTMobilenet(model.undesired_content_model, arguments.data_path, backbone_strategy=arguments.backbone_train)
        elif arguments.backbone == 'inception':
            from core.undesired_content_inception import UndesiredContentInception
            from mainscripts.fine_tuning.inception import FTInception

            model = UndesiredContentInception()
            model = FTInception(model.undesired_content_model, arguments.data_path, backbone_strategy=arguments.backbone_train)
        elif arguments.backbone == 'efficientnet':
            from core.undesired_content_efficientnet import UndesiredContentEfficientNet
            from mainscripts.fine_tuning.efficientnet import FTEfficientNet

            model = UndesiredContentEfficientNet()
            model = FTEfficientNet(model.undesired_content_model, arguments.data_path, backbone_strategy=arguments.backbone_train)

        if arguments.evaluate:
            model.evaluate_on_test_set()
        elif arguments.write_pred:
            model.write_logits_to_csv_on_test()
        else:
            model.training()

    @classmethod
    def _parser_training(cls):
        cls.p = cls.subparsers.add_parser("finetune", help="finetune a model")
        cls.p.add_argument('--evaluate', '-e', required=False, default=False, action="store_true", dest="evaluate",
                           help="if selected evaluates best model so far on the test data")
        cls.p.add_argument('--write-pred', '-wp', required=False, default=False, action="store_true", dest="write_pred",
                           help="if selected evaluates best model so far on the test data")
        cls.p.add_argument('--data-path', '-dp', required=True, dest="data_path",
                           help="data path contains images for training.")
        cls.p.add_argument('--backbone-train', '-bt', required=True, dest="backbone_train",
                           choices=['full', 'partial', 'freeze'],
                           help="backbone training strategy.")
        cls.p.add_argument('--backbone', '-b', required=True, dest="backbone",
                           choices=['mobilenet', 'inception', 'efficientnet'],
                           help="backbone architecture.")

        cls.p.set_defaults(func=cls._process_training)

    @staticmethod
    def _process_filter(arguments):
        from mainscripts.filter import UndesiredContentFilter
        from libs.video_editor import extract_frames
        extract_frames(arguments.videos_path, int(os.environ['FRAME_N_SIZE']))
        UndesiredContentFilter().run(os.environ['FRAMES_DIR'])

    @classmethod
    def _parser_filter(cls):
        cls.p = cls.subparsers.add_parser("filter", help="Filter a video if has a undesired content")

        cls.p.add_argument('--videos-path', '-vp', required=True, dest="videos_path",
                           help="An input video path for a mp4 file to process.")

        cls.p.set_defaults(func=cls._process_filter)

    @staticmethod
    def _process_simulation(arguments):
        from mainscripts.simulation import Simulator

        if arguments.find_threshold:
            Simulator().run_threshold_finder_for_images()
        else:
            if arguments.phase == 'phase_1' or arguments.phase == 'both':
                Simulator(arguments.phase_1_csv).run_phase_1()
            if arguments.phase == 'phase_2' or arguments.phase == 'both':
                Simulator(arguments.phase_1_csv).run_phase_2(arguments.method)
            else:
                raise AttributeError('Unknown phase option')

    @classmethod
    def _parser_simulation(cls):
        cls.p = cls.subparsers.add_parser("simulation", help="simulation for calculating best parameters for "
                                                             "detection system along with prediction of ai models. ")

        cls.p.add_argument('--find-threshold', '-ft', required=False, default=False, action="store_true", dest="find_threshold",
                           help="finds threshold do not need other arguments if selected")
        cls.p.add_argument('--phase', '-p', required=True, dest="phase", choices=['phase_1', 'phase_2', 'both'],
                           help="An input video directory containing the mp4 files to process.")
        cls.p.add_argument('--method', '-m', required=True, dest="method", choices=['method_1', 'method_2'],
                           help="parameter search in phase 2 will be done for selected method.")
        cls.p.add_argument('--phase-1-csv-path', '-pcv', required=True, dest="phase_1_csv",
                           help="csv path for phase 1 which holds predictions from ai models.")

        cls.p.set_defaults(func=cls._process_simulation)

    @staticmethod
    def _process_evaluation(arguments):
        from mainscripts.simulation import Simulator
        Simulator().run_evaluation()

    @classmethod
    def _parser_evaluation(cls):
        cls.p = cls.subparsers.add_parser("evaluation", help="evaluates given parameters on videos in the video folder ")

        cls.p.set_defaults(func=cls._process_evaluation)

    @classmethod
    def _wrong_args(cls, arguments):
        cls.parser.print_help()
        exit(0)

    @classmethod
    def _parser_wrong_args(cls):
        cls.parser.set_defaults(func=cls._wrong_args)

    @classmethod
    def run(cls):
        cls._parser_simulation()
        cls._parser_filter()
        cls._parser_data_generate()
        cls._parser_training()
        cls._parser_wrong_args()
        cls._parser_evaluation()

        arguments = cls.parser.parse_args()
        arguments.func(arguments)


if __name__ == '__main__':
    MainScript.run()
