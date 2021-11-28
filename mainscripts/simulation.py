import os
from os import path
import ast
from mainscripts.filter import load_images
from libs.video_editor import extract_frames, key_frames
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from tqdm.auto import tqdm
import csv
import numpy as np
import itertools


class Simulator:
    def __init__(self, phase_1_csv_path=''):

        self.phase_1_csv_path = phase_1_csv_path
        self.phase_2_csv_path = path.join(
            path.dirname(phase_1_csv_path),
            f"grid_search_{path.basename(phase_1_csv_path)}")

        self.video_list = os.listdir(os.environ['VIDEO_DIR'])

        self.frame_n_size = 1  # as default we search for every image

        self.s_space = None
        self.max_iteration = None

    def define_search_space_method_1(self):
        # first model
        alpha_undesired_content_list = [0.60,0.75]
        lambda_undesired_content_list = [0.60,0.70,0.93, 0.95]

        # second model
        gamma_undesired_content_list = [0.85]

        # combined
        max_frame_list = [50]
        nth_frame_list = [12]
        review_perc_list = [0.95]
        beta_perc_list = [0.08]

        combination = [alpha_undesired_content_list,
                       lambda_undesired_content_list,
                       gamma_undesired_content_list,
                       max_frame_list,
                       nth_frame_list,
                       beta_perc_list,
                       review_perc_list]

        # cartesian search space
        self.s_space = itertools.product(*combination)
        self.max_iteration = np.product([len(x) for x in combination])

    def define_search_space_method_2(self):
        # first model
        alpha_undesired_content_list = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        # second model
        lambda_undesired_content_list = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        # combined
        max_frame_list = [20, 30, 40, 50]
        nth_frame_list = [3, 6, 8, 12]
        review_perc_list = [0.75, 0.80, 0.85, 0.90, 0.95]
        beta_perc_list = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08]

        combination = [alpha_undesired_content_list,
                       lambda_undesired_content_list,
                       max_frame_list,
                       nth_frame_list,
                       beta_perc_list,
                       review_perc_list]

        # cartesian search space
        self.s_space = itertools.product(*combination)
        self.max_iteration = np.product([len(x) for x in combination])

    def run_phase_1(self):
        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        from core.finetuned_mobilenet import UndesiredContentMobileNet
        from core.finetuned_inception import UndesiredContentInception
        from core.finetuned_efficientnet import UndesiredContentEfficientNet

        models = [UndesiredContentMobileNet(), UndesiredContentEfficientNet()]

        simulation_data = []

        with open(self.phase_1_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_i", "label",'result_mob',"result_eff", "i_frames", "frame_len"])

        for idx, video_i in tqdm(enumerate(self.video_list), total=len(self.video_list), desc="video_loop", position=0,
                                 leave=False):
            i_frames = key_frames(os.path.join(os.environ['VIDEO_DIR'], video_i))
            extract_frames(os.path.join(os.environ['VIDEO_DIR'], video_i), frame_n_size=self.frame_n_size)
            # key frame > some n can be considered for check out while sending video to user.

            images, img_paths = load_images(os.environ['FRAMES_DIR'])

            frame_len = images.shape[0]

            model_results = []

            for i, model_i in enumerate(models):
                results = []
                for frame_i in tqdm(images, total=images.shape[0], desc=f"model_{i}_frame_loop", leave=False, position=i+1):
                    pred_i = model_i.predict(frame_i)
                    results.append(float(pred_i.round(5)))
                model_results.append(results)

            if video_i[0] == 's' or video_i[0] == 'f':
                label = 0
            elif video_i[0] == "n":
                label = 1
            else:
                raise ValueError('wrong file starting character')

            simulation_data.append([video_i] + [label] + model_results + [i_frames] + [frame_len])

            if idx % 10 == 0:
                with open(self.phase_1_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(simulation_data)

                simulation_data = []

        with open(self.phase_1_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(simulation_data)

    def phase_2_method_1(self):

        from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

        self.define_search_space_method_1()

        np.random.seed(100)

        with open(self.phase_1_csv_path, "r", newline="") as csvfile:
            readCSV = csv.reader(csvfile)
            data_undesired_content = []
            for row in readCSV:
                data_undesired_content.append(row)
            print(data_undesired_content.pop(0))

        result = []

        for elements in tqdm(self.s_space, total=self.max_iteration, desc="parameter_loop", position=0, leave=False):

            alpha_undesired_content, lambda_undesired_content, gamma_undesired_content, max_frame, nth_frame, beta_perc, review_perc = elements
            pred_label = []
            true_label = []
            review_label = []

            for video_i, label, first_preds, pred_d, i_frames, frame_len in tqdm(data_undesired_content, total=len(data_undesired_content),
                                                                                 desc="video_loop", position=1,
                                                                                 leave=False):

                label, first_preds, second_preds, i_frames, frame_len = [ast.literal_eval(z) for z in
                                                                         [label, first_preds, pred_d, i_frames,
                                                                          frame_len]]

                # 1 means 3th, 2 means 6th
                selections = np.array(range(1, (frame_len // nth_frame) + 1)) * nth_frame

                if len(i_frames) > max_frame:
                    frames_to_check = np.sort(np.random.choice(i_frames, max_frame, replace=False))
                elif len(i_frames) == max_frame:
                    frames_to_check = i_frames
                elif len(i_frames) + len(selections) > max_frame:
                    diff_frames = np.setdiff1d(selections, i_frames)
                    diff_selected = np.sort(
                        np.random.choice(diff_frames, min(max_frame, len(diff_frames)) - len(i_frames), replace=False))
                    frames_to_check = np.sort(np.concatenate([i_frames, diff_selected]))
                else:
                    diff_frames = np.setdiff1d(selections, i_frames)
                    frames_to_check = np.sort(np.concatenate([i_frames, diff_frames]))

                frames_to_check = frames_to_check - 1

                first_preds = np.array(first_preds)
                second_preds = np.array(second_preds)

                first_preds = first_preds[frames_to_check, ...]
                second_preds = second_preds[frames_to_check, ...]

                undesired_content_idx = first_preds >= lambda_undesired_content
                scores = first_preds[undesired_content_idx].tolist()
                count = sum(undesired_content_idx)
                post_result = 0
                percentage = 0.0

                if (count / frame_len) < beta_perc:
                    dangers_idx = np.logical_and(np.array(first_preds) >= alpha_undesired_content,
                                                 np.array(first_preds) < lambda_undesired_content)
                    dangers_ratio = second_preds[dangers_idx].tolist()
                    dangers_ratio_first = first_preds[dangers_idx].tolist()
                else:
                    post_result = 1
                    dangers_ratio = None
                    dangers_idx = None
                    percentage = first_preds[undesired_content_idx].mean()

                if dangers_ratio:
                    dangers = dangers_idx.tolist()

                    if len(dangers) >= 2:
                        dangers = np.array(dangers_ratio_first).argsort()[::-1]

                    for im_i in dangers:
                        pred = dangers_ratio[im_i]
                        score = pred if pred >= gamma_undesired_content else None

                        if score:
                            count = count + 1
                            scores.append(score)
                            scores.append(dangers_ratio_first[im_i])

                        if (count / frame_len) >= beta_perc:
                            post_result = 1
                            percentage = np.array(scores).mean()
                            break

                if percentage:
                    review = 1 if percentage > review_perc else 0
                else:
                    review = 0

                pred_label.append(post_result)
                true_label.append(label)
                review_label.append(review)
            undesired_content_lb = np.array(pred_label) == 1
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            review_label = np.array(review_label)

            result.append([(alpha_undesired_content, lambda_undesired_content, gamma_undesired_content, max_frame, nth_frame, beta_perc, review_perc),
                           f1_score(true_label, pred_label), accuracy_score(true_label, pred_label),
                           tuple(confusion_matrix(true_label, pred_label).ravel()),
                           tuple(confusion_matrix(true_label[undesired_content_lb], review_label[undesired_content_lb]).ravel())])

        # confusion matrix sequence tn,fp,fn,tp

        with open(self.phase_2_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result)

    def phase_2_method_2(self):

        from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
        import multiprocessing as mp

        # for each video for each frame and for each model

        np.random.seed(100)
        self.define_search_space_method_2()

        with open(self.phase_1_csv_path, "r", newline="") as csvfile:
            readCSV = csv.reader(csvfile)
            data_undesired_content = []
            count = 0
            for row in readCSV:
                if count != 0:
                    row[1] = ast.literal_eval(row[1])
                    row[2] = ast.literal_eval(row[2])
                    row[3] = ast.literal_eval(row[3])
                    row[4] = ast.literal_eval(row[4])
                    row[5] = ast.literal_eval(row[5])
                    data_undesired_content.append(row)
                count = count + 1
            print("video count:", count)

        data_undesired_content = np.array(data_undesired_content)

        result = []

        global alpha_undesired_content
        global lambda_undesired_content
        global max_frame
        global nth_frame
        global beta_perc
        global review_perc

        for elements in tqdm(self.s_space, total=self.max_iteration, desc="parameter_loop", position=0, leave=False):
            alpha_undesired_content, lambda_undesired_content, max_frame, nth_frame, beta_perc, review_perc = elements

            limit = mp.cpu_count()
            pool = mp.Pool(limit)

            output = pool.map(self.undesired_content_filter_single_video, data_undesired_content)
            pool.close()
            pool.join()

            output = np.array(output)
            pred_label, true_label, review_label = [np.squeeze(x) for x in np.split(output, 3, axis=1, )]

            undesired_content_lb = np.array(pred_label) == 1

            result.append([(alpha_undesired_content, lambda_undesired_content, max_frame, nth_frame, beta_perc, review_perc),
                           f1_score(true_label, pred_label), accuracy_score(true_label, pred_label),
                           tuple(confusion_matrix(true_label, pred_label).ravel()),
                           tuple(confusion_matrix(true_label[undesired_content_lb], review_label[undesired_content_lb]).ravel())])

        with open(self.phase_2_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result)

    def run_phase_2(self, method):
        if method == 'method_1':
            self.phase_2_method_1()
        elif method == "method_2":
            self.phase_2_method_2()
        else:
            raise AttributeError('Unknown Error')

    @staticmethod
    def undesired_content_filter_single_video(x):

        video_i, label, first_preds, second_preds, i_frames, frame_len = x

        selections = np.array(range(1, (frame_len // nth_frame) + 1)) * nth_frame

        if len(i_frames) > max_frame:
            frames_to_check = np.sort(np.random.choice(i_frames, max_frame, replace=False))
        elif len(i_frames) == max_frame:
            frames_to_check = np.array(i_frames)
        elif len(i_frames) + len(selections) > max_frame:
            diff_frames = np.setdiff1d(selections, i_frames)
            diff_selected = np.sort(
                np.random.choice(diff_frames, max(min(max_frame, len(diff_frames)) - len(i_frames), 0),
                                 replace=False))
            frames_to_check = np.sort(np.concatenate([i_frames, diff_selected]))
        else:
            diff_frames = np.setdiff1d(selections, i_frames)
            frames_to_check = np.sort(np.concatenate([i_frames, diff_frames]))

        frames_to_check = frames_to_check - 1

        first_preds = np.array(first_preds).astype('float32')
        second_preds = np.array(second_preds).astype('float32')

        first_preds = first_preds[frames_to_check]
        second_preds = second_preds[frames_to_check]

        undesired_content1_idx = np.where(first_preds >= lambda_undesired_content, 10, 1)

        undesired_content2_idx = np.where(second_preds >= alpha_undesired_content, 10, 1)

        undesired_content_idx = undesired_content1_idx * undesired_content2_idx

        perc = np.mean(undesired_content_idx) / 100

        if perc > beta_perc:
            post_result = 1
            percentage = np.mean(
                np.concatenate([first_preds[first_preds >= lambda_undesired_content], second_preds[second_preds >= alpha_undesired_content]]))
        else:
            post_result = 0
            percentage = None

        if percentage:
            review = 1 if percentage > review_perc else 0
        else:
            review = 0

        return post_result, label, review

    def run_evaluation(self):

        alpha_undesired_content = 0.8
        lambda_undesired_content = 0.7
        max_frame = 30
        nth_frame = 12
        review_perc = 0.85
        beta_perc = 0.03

        import tensorflow as tf
        from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        from core.finetuned_mobilenet import UndesiredContentMobileNet
        from core.finetuned_efficientnet import UndesiredContentEfficientNet

        mobilenet = UndesiredContentMobileNet()
        effnet = UndesiredContentEfficientNet()

        pred_label = []
        true_label = []
        review_label = []

        for idx, video_i in tqdm(enumerate(self.video_list), total=len(self.video_list), desc="video_loop", position=0,
                                 leave=False):

            i_frames = key_frames(os.path.join(os.environ['VIDEO_DIR'], video_i))
            extract_frames(os.path.join(os.environ['VIDEO_DIR'], video_i), frame_n_size=self.frame_n_size)
            # key frame > some n can be considered for check out while sending video to user.

            images, img_paths = load_images(os.environ['FRAMES_DIR'])

            frame_len = images.shape[0]

            selections = np.array(range(1, (frame_len // nth_frame) + 1)) * nth_frame

            if len(i_frames) > max_frame:
                frames_to_check = np.sort(np.random.choice(i_frames, max_frame, replace=False))
            elif len(i_frames) == max_frame:
                frames_to_check = i_frames
            elif len(i_frames) + len(selections) > max_frame:
                diff_frames = np.setdiff1d(selections, i_frames)
                diff_selected = np.sort(
                    np.random.choice(diff_frames, min(max_frame, len(diff_frames)) - len(i_frames), replace=False))
                frames_to_check = np.sort(np.concatenate([i_frames, diff_selected]))
            else:
                diff_frames = np.setdiff1d(selections, i_frames)
                frames_to_check = np.sort(np.concatenate([i_frames, diff_frames]))

            frames_to_check = frames_to_check - 1

            # filter images
            try:
                images = images[frames_to_check,...]
                img_paths = img_paths[frames_to_check]
            except:
                continue

            first_preds = []

            for img_i in images:
                pred_i = mobilenet.predict(img_i)
                first_preds.append(pred_i)

            second_preds = []

            for img_i in images:
                pred_i = effnet.predict(img_i)
                second_preds.append(pred_i)

            if video_i[0] == 's' or video_i[0] == 'f':
                label = 0
            elif video_i[0] == "n":
                label = 1
            else:
                raise ValueError('wrong file starting character')

            first_preds = np.array(first_preds).astype('float32')
            second_preds = np.array(second_preds).astype('float32')

            undesired_content1_idx = np.where(first_preds >= lambda_undesired_content, 10, 1)

            undesired_content2_idx = np.where(second_preds >= alpha_undesired_content, 10, 1)

            undesired_content_idx = undesired_content1_idx * undesired_content2_idx

            perc = np.mean(undesired_content_idx) / 100.0

            if perc > beta_perc:
                post_result = 1
                percentage = np.mean(
                    np.concatenate([first_preds[first_preds >= lambda_undesired_content], second_preds[second_preds >= alpha_undesired_content]]))
            else:
                post_result = 0
                percentage = 0.

            if percentage>0.:
                review = 1 if percentage > review_perc else 0
            else:
                review = 0


            true_label.append(label)
            pred_label.append(post_result)
            review_label.append(review)

            print(video_i,label,post_result,review,round(perc,6),round(float(percentage),6))
        true_label = np.array(true_label)
        pred_label = np.array(pred_label)
        review_label = np.array(review_label)
        undesired_content_lb = pred_label == 1

        print(f1_score(true_label, pred_label),
              accuracy_score(true_label, pred_label),
              tuple(confusion_matrix(true_label, pred_label).ravel()),
              tuple(confusion_matrix(true_label[undesired_content_lb], review_label[undesired_content_lb]).ravel()))

    def run_threshold_finder_for_images(self):

        from tensorflow_addons.metrics import F1Score

        # read mobilenet and efficient net results
        # there is no header
        data_1 = 'data/predictions_mobile.csv'
        data_2 = 'data/predictions_efficientnet.csv'

        with open(data_1, "r", newline="") as csvfile:
            readCSV = csv.reader(csvfile)
            data_mobile = []
            for row in readCSV:
                data_mobile.append(row)

        with open(data_2, "r", newline="") as csvfile:
            readCSV = csv.reader(csvfile)
            data_efficient = []
            for row in readCSV:
                data_efficient.append(row)

        data_mobile = np.array(data_mobile, dtype=np.float32)
        data_efficient = np.array(data_efficient, dtype=np.float32)

        metric = F1Score(num_classes=1, threshold=0.50, average='macro', name='f1_score', dtype='float32')
        metric.update_state(np.expand_dims(data_mobile[:,1],-1),np.expand_dims(data_mobile[:,2],-1))
        metric.result()

        metric = F1Score(num_classes=1, threshold=0.50, average='macro', name='f1_score', dtype='float32')
        metric.update_state(np.expand_dims(data_efficient[:,1],-1),np.expand_dims(data_efficient[:,2],-1))
        metric.result()

        logit = (np.expand_dims(data_mobile[:,2],-1) + 2*np.expand_dims(data_efficient[:,2],-1)) / 3

        metric = F1Score(num_classes=1, threshold=0.30, average='macro', name='f1_score', dtype='float32')
        metric.update_state(np.expand_dims(data_efficient[:,1],-1),logit)
        metric.result()

        undesired_content_idx = np.where(logit >= 0.45, 1.0, 0.0)
        f1_score(np.expand_dims(data_efficient[:,1],-1), undesired_content_idx)
        tuple(confusion_matrix(np.expand_dims(data_efficient[:,1],-1), undesired_content_idx).ravel())












