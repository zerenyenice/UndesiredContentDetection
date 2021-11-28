from libs.tf_recorder import TfRecorder
from libs.augmentor import Augmentor
from tensorflow_addons.metrics import F1Score, MatthewsCorrelationCoefficient as MCC, MultiLabelConfusionMatrix as CM
import tensorflow as tf
import os
import datetime
import csv


class FineTuningBase(TfRecorder, Augmentor):
    """
    Data needs to be start letters with 's' and 'n' respectively
    safe and not safe and stored under the directory given as data_dir.
    Tf records will be created for the given data.
    """

    def __init__(self, model, data_dir, image_size=(296, 224), backbone_strategy='freeze'):
        '''

        :param model: keras model architecture
        :param data_dir: directory of the data
        :param image_size: image size for training
        :param backbone_strategy: needs to be selected one of 'freeze', 'train', 'finetune'
        '''
        self.tf_record_training = 'training_data.tfrec'
        self.tf_record_validation = 'validation_data.tfrec'
        self.tf_record_test = 'test_data.tfrec'
        self.label_csv = f'{data_dir}.csv'
        self.tf_record_training_path = os.path.join(os.path.dirname(data_dir), self.tf_record_training)
        self.tf_record_validation_path = os.path.join(os.path.dirname(data_dir), self.tf_record_validation)
        self.tf_record_test_path = os.path.join(os.path.dirname(data_dir), self.tf_record_test)
        self.model = model

        self.optimizer = None
        self.loss = None
        self.checkpoint_manager = None
        self.data_dir = data_dir
        self.image_size = image_size
        TfRecorder.__init__(self)
        Augmentor.__init__(self)
        self.init_augmentation_pipeline()
        self.train_config = None
        self.backbone_strategy = backbone_strategy
        self.unlock_after = None
        self.freeze_batch_norm_layers = False
        self.f1_score_func = F1Score(num_classes=1, threshold=0.5)
        self.sample_weight_coef = 0.0 # means no sample weights

        self.backup_dir = None
        self.checkpoint = None
        self.checkpoint_manager = None
        self.global_epoch = None
        self.loss_metrics = None
        self.best_f1_score = None

    def get_train_config(self):
        raise NotImplementedError

    def use_adabelief_optimizer(self):
        from adabelief_tf import AdaBeliefOptimizer
        return AdaBeliefOptimizer(learning_rate=1e-3,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  weight_decay=1e-2,
                                  amsgrad=False,
                                  epsilon=1e-8,
                                  rectify=False)

    @staticmethod
    def use_lookahead(optimizer):
        from tensorflow_addons.optimizers import Lookahead
        return Lookahead(optimizer)


    def read_csv_label_data(self):

        with open(self.label_csv, mode='r') as infile:
            reader = csv.reader(infile)
            mydict = {rows[0]: rows[1] for rows in reader}

        return mydict

    def initialize_tf_record_creation(self):
        labels = self.read_csv_label_data()
        self.create_tfrecord(os.path.join(self.data_dir, 'train_set'), self.tf_record_training_path, labels)
        self.create_tfrecord(os.path.join(self.data_dir, 'validate_set'), self.tf_record_validation_path, labels)
        self.create_tfrecord(os.path.join(self.data_dir, 'test_set'), self.tf_record_test_path, labels)

    def _train_validation_generator(self):
        ds_train = self.get_tf_record(self.tf_record_training_path)
        ds_validation = self.get_tf_record(self.tf_record_validation_path)
        t_s = self.get_data_size(ds_train)
        v_s = self.get_data_size(ds_validation)

        ds_train = ds_train.cache().shuffle(t_s)
        ds_validation = ds_validation.cache().shuffle(v_s)

        ds_train = self.get_data(ds_train)
        ds_validation = self.get_data(ds_validation)

        return ds_train, ds_validation

    def _test_generator(self):
        ds = self.get_tf_record(self.tf_record_test_path)

        test_list_ds = self.get_data(ds)

        return test_list_ds

    def _wrapped_augmentation(self):

        def apply_augmentation_to_data_pipe(image,label):
            img_dtype = image.dtype
            tf_image = tf.numpy_function(self.single_image_augmentation,
                                         [image],
                                         img_dtype)
            tf_image.set_shape([None,None,3])
            return tf_image, label

        return apply_augmentation_to_data_pipe

    def _apply_augmentation_to_data_pipe(self, data: tf.data.Dataset):
        return data.map(self._wrapped_augmentation(), num_parallel_calls=self.AUTO)

    def _wrapped_resize_and_norm(self):

        def resize_and_norm_data(image,label):
            image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        return resize_and_norm_data

    def _apply_resize_and_norm_to_data_pipe(self, data: tf.data.Dataset):
        return data.map(self._wrapped_resize_and_norm(), num_parallel_calls=self.AUTO)

    @staticmethod
    def print_metrics_training(epoch, loss, acc, f1, mcc, conf_mat, p_param):
        print(f"|{datetime.datetime.now().strftime('%Y/%m/%d-%H:%M:%S')}|"
              f"|epoch:{epoch:05}|"
              f"|loss: {loss:.6f}|"
              f"|acc: {acc:.4f}|"
              f"|f1: {f1:.4f}|"
              f"|mcc: {mcc:.4f}|"
              f"|conf_mat: {conf_mat}|", end=p_param)


    @staticmethod
    def print_metrics_evaluation(loss, acc, f1, mcc, conf_mat, p_param):
        print(f"|val_loss: {loss:.6f}|\n"
              f"|val_acc: {acc:.4f}|\n"
              f"|val_f1: {f1:.4f}|\n"
              f"|val_mcc: {mcc:.4f}|\n"
              f"|val_conf_mat: {conf_mat}|", end=p_param)

    def get_training_data_pipeline(self):
        train_ds, val_ds = self._train_validation_generator()
        train_ds = self._apply_augmentation_to_data_pipe(train_ds)

        train_ds = self._apply_resize_and_norm_to_data_pipe(train_ds)
        val_ds = self._apply_resize_and_norm_to_data_pipe(val_ds)

        train_ds = train_ds.batch(self.train_config['batch_size']).prefetch(self.AUTO)
        val_ds = val_ds.batch(self.train_config['batch_size']).prefetch(self.AUTO)
        return train_ds, val_ds

    def get_test_data_pipeline(self):
        test_ds = self._test_generator()
        test_ds = self._apply_resize_and_norm_to_data_pipe(test_ds)
        test_ds = test_ds.batch(1).prefetch(self.AUTO)

        return test_ds

    def create_checkpoint_system(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model,
                                              optimizer=self.optimizer,
                                              global_epoch=self.global_epoch,
                                              best_f1_score=self.best_f1_score)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.backup_dir, max_to_keep=6)

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Latest checkpoint restored!! --> {self.checkpoint_manager.latest_checkpoint}. it will continue to training')
        else:
            print('There is no checkpoint, new training will be initialized')

    def reset_loss_holders(self):
        if self.loss_metrics is None:
            raise ValueError('Firstly create loss metrics')

        for _, item in self.loss_metrics.items():
            item.reset_states()

    def initialize_metrics(self):

        self.loss_metrics = {
            "loss": tf.keras.metrics.BinaryCrossentropy(label_smoothing=self.train_config['label_smoothing'],name='binary_crossentropy', dtype='float32'),
            "accuracy": tf.keras.metrics.Mean(name='accuracy', dtype='float32'),
            "f1_score": F1Score(num_classes=1, threshold=0.5, average='macro',name='f1_score', dtype='float32'),
            "MCC": MCC(num_classes=1, name='MCC', dtype='float32'),
            "conf_mat": CM(num_classes=1, name='CM'),
            "val_loss": tf.keras.metrics.BinaryCrossentropy(label_smoothing=self.train_config['label_smoothing'],name='val_binary_crossentropy', dtype='float32'),
            "val_accuracy": tf.keras.metrics.Mean(name='val_accuracy', dtype='float32'),
            "val_f1_score": F1Score(num_classes=1, threshold=0.5, average='macro',name='val_f1_score', dtype='float32'),
            "val_MCC": MCC(num_classes=1,name='validation_MCC', dtype='float32'),
            "val_conf_mat": CM(num_classes=1, name='validation_CM'),
            "test_accuracy": tf.keras.metrics.Mean(name='test_accuracy', dtype='float32'),
            "test_f1_score": F1Score(num_classes=1, threshold=0.5, average='macro',name='test_f1_score', dtype='float32'),
            "test_MCC": MCC(num_classes=1, name='test_MCC', dtype='float32'),
            "test_conf_mat": CM(num_classes=1, name='test_CM')
        }

    def get_loss_function(self):

        #def loss_fn(y_true, y_pred):
        #    return tf.metrics.binary_crossentropy(y_true,
        #                                          y_pred,
        #                                          label_smoothing=self.train_config['label_smoothing'])

        return tf.losses.BinaryCrossentropy(label_smoothing=self.train_config['label_smoothing'])

    def initialize_training_elements(self):
        self.global_epoch = tf.Variable(tf.constant(0, dtype='int64'), trainable=False, name='global_epoch')
        self.best_f1_score = tf.Variable(tf.constant(0., dtype='float32'), trainable=False, name='best_f1_score')
        self.loss = self.get_loss_function()
        # self.optimizer = tf.optimizers.RMSprop(learning_rate=self.train_config['learning_rate'],
        #                                        epsilon=self.train_config['opt_epsilon'],
        #                                        momentum=self.train_config['momentum'],
        #                                        decay=self.train_config['decay'])

        self.optimizer = self.use_adabelief_optimizer()

    def convert_model_for_finetune(self):
        raise NotImplementedError

    @tf.function
    def train_step(self,batch_image,batch_label):
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_image, training=True)
            if self.sample_weight_coef > 0:
                weights = ((batch_label * self.sample_weight_coef) + 1) / (self.sample_weight_coef+1)
            else:
                weights = (((1.0-batch_label) * -1*self.sample_weight_coef) + 1) / (-1*self.sample_weight_coef + 1)
            loss = tf.reduce_mean(self.loss(batch_label, y_pred, weights))

            if self.model.losses:
                loss += tf.add_n(self.model.losses)

            acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(batch_label, y_pred))
            self.loss_metrics['f1_score'].update_state(batch_label, y_pred)
            mcc_y_pred = y_pred >= 0.5
            mcc_y_pred = tf.cast(mcc_y_pred, tf.float32)

            self.loss_metrics['MCC'].update_state(batch_label, mcc_y_pred)
            self.loss_metrics['conf_mat'].update_state(batch_label, mcc_y_pred)
            self.loss_metrics['loss'].update_state(batch_label,y_pred,weights)
            self.loss_metrics['accuracy'].update_state(acc)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss, acc

    @tf.function
    def validate_step(self, batch_image, batch_label):

        y_pred = self.model(batch_image, training=False)
        if self.sample_weight_coef > 0:
            weights = ((batch_label * self.sample_weight_coef) + 1) / (self.sample_weight_coef + 1)
        else:
            weights = (((1.0 - batch_label) * -1 * self.sample_weight_coef) + 1) / (-1 * self.sample_weight_coef + 1)

        acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(batch_label, y_pred))
        self.loss_metrics['val_f1_score'].update_state(batch_label, y_pred)
        mcc_y_pred = y_pred >= 0.5
        mcc_y_pred = tf.cast(mcc_y_pred, tf.float32)
        self.loss_metrics['val_MCC'].update_state(batch_label, mcc_y_pred)
        self.loss_metrics['val_conf_mat'].update_state(batch_label, mcc_y_pred)
        self.loss_metrics['val_loss'].update_state(batch_label,y_pred,weights)
        self.loss_metrics['val_accuracy'].update_state(acc)

        return acc

    @tf.function
    def test_step(self, batch_image, batch_label):

        y_pred = self.model(batch_image, training=False)

        acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(batch_label, y_pred))

        mcc_y_pred = y_pred >= 0.5
        mcc_y_pred = tf.cast(mcc_y_pred, tf.float32)

        self.loss_metrics['test_accuracy'].update_state(acc)
        self.loss_metrics['test_f1_score'].update_state(batch_label, y_pred)
        self.loss_metrics['test_MCC'].update_state(batch_label, mcc_y_pred)
        self.loss_metrics['test_conf_mat'].update_state(batch_label, mcc_y_pred)

        return acc

    @tf.function
    def test_step_return_probability(self, batch_image):

        y_pred = self.model(batch_image, training=False)

        return y_pred

    def training(self):
        self.initialize_tf_record_creation()
        train_ds, validation_ds = self.get_training_data_pipeline()
        self.convert_model_for_finetune()
        self.initialize_training_elements()
        self.initialize_metrics()
        self.create_checkpoint_system()
        print('best f1 score achieved so far: ',self.best_f1_score)
        print('starting: ', self.global_epoch)

        print("LAYERS TO TRAIN:")
        for i in self.model.trainable_weights:
            print(i.name)

        for epoch_i in range(self.global_epoch.numpy(), self.train_config['epoch']):

            try:

                for step, (images, labels) in enumerate(train_ds):

                    step_loss, step_acc = self.train_step(images, labels)
                    if step % 1 == 0:
                        self.print_metrics_training(epoch_i,
                                                    step_loss,
                                                    step_acc,
                                                    tf.squeeze(self.loss_metrics['f1_score'].result()).numpy(),
                                                    tf.squeeze(self.loss_metrics['MCC'].result()).numpy(),
                                                    tuple(self.loss_metrics['conf_mat'].result().numpy().ravel()), '\r')

                self.print_metrics_training(epoch_i,
                                            self.loss_metrics['loss'].result().numpy(),
                                            self.loss_metrics['accuracy'].result().numpy(),
                                            tf.squeeze(self.loss_metrics['f1_score'].result()).numpy(),
                                            tf.squeeze(self.loss_metrics['MCC'].result()).numpy(),
                                            tuple(self.loss_metrics['conf_mat'].result().numpy().ravel()), '\n')
                for images, labels in validation_ds:

                    _ = self.validate_step(images, labels)

                self.print_metrics_evaluation(
                    self.loss_metrics['val_loss'].result().numpy(),
                    self.loss_metrics['val_accuracy'].result().numpy(),
                    tf.squeeze(self.loss_metrics['val_f1_score'].result()).numpy(),
                    tf.squeeze(self.loss_metrics['val_MCC'].result()).numpy(),
                    tuple(self.loss_metrics['val_conf_mat'].result().numpy().ravel()),
                    '\n')

                val_f1 = tf.squeeze(self.loss_metrics['val_f1_score'].result()).numpy()

                if val_f1 > self.best_f1_score:
                    print(val_f1, ">", self.best_f1_score, " saving best model")
                    self.best_f1_score.assign(val_f1)
                    self.checkpoint.write(os.path.join(self.backup_dir,'best_acc'))
                    self.model.save(os.path.join(self.backup_dir,'best_acc.h5'))

                self.reset_loss_holders()

                if epoch_i % self.train_config['backup_cycle'] == 0:
                    self.checkpoint_manager.save(epoch_i)

            except KeyboardInterrupt:

                print('training interrupted')
                self.checkpoint_manager.save(1)
                print('training ended with user interruption')
                break

    def evaluate_on_test_set(self):
        self.initialize_tf_record_creation()
        self.initialize_metrics()
        test_ds = self.get_test_data_pipeline()

        self.model = tf.keras.models.load_model(os.path.join(self.backup_dir,'best_acc.h5'))

        for images, labels in test_ds:
            _ = self.test_step(images, labels)

        self.print_metrics_evaluation(
            0.0,
            self.loss_metrics['test_accuracy'].result().numpy(),
            tf.squeeze(self.loss_metrics['test_f1_score'].result()).numpy(),
            tf.squeeze(self.loss_metrics['test_MCC'].result()).numpy(),
            tuple(self.loss_metrics['test_conf_mat'].result().numpy().ravel()),
            '\n')

    def write_logits_to_csv_on_test(self):
        self.initialize_tf_record_creation()
        self.initialize_metrics()
        test_ds = self.get_test_data_pipeline()

        self.model = tf.keras.models.load_model(os.path.join(self.backup_dir,'best_acc.h5'))

        with open('data/predictions.csv', "a", newline="") as f:
            writer = csv.writer(f)

            for i, (images, labels) in enumerate(test_ds):
                y_pred = self.test_step_return_probability(images)
                writer.writerow([i,float(labels),float(y_pred)])



class FineTuningBaseSep(FineTuningBase):

    def __init__(self, model, data_dir, image_size=(296, 224), backbone_strategy='freeze'):
        '''

        :param model: keras model architecture
        :param data_dir: directory of the data
        :param image_size: image size for training
        :param backbone_strategy: needs to be selected one of 'freeze', 'train', 'finetune'
        '''
        FineTuningBase.__init__(self,model,data_dir,image_size,backbone_strategy)
        #TfRecorder.__init__(self)
        #Augmentor.__init__(self)
        self.tf_record_training_safe = 'training_data_safe.tfrec'
        self.tf_record_training_undesired_content = 'training_data_undesired_content.tfrec'

        self.tf_record_training_safe_path = os.path.join(os.path.dirname(data_dir), self.tf_record_training_safe)
        self.tf_record_training_undesired_content_path = os.path.join(os.path.dirname(data_dir), self.tf_record_training_undesired_content)

    def initialize_tf_record_creation(self):
        labels = self.read_csv_label_data()
        self.create_tfrecord_seperated_files(os.path.join(self.data_dir, 'train_set'), self.tf_record_training_safe_path, labels, '0')
        self.create_tfrecord_seperated_files(os.path.join(self.data_dir, 'train_set'),
                                             self.tf_record_training_undesired_content_path, labels, '1')
        self.create_tfrecord(os.path.join(self.data_dir, 'validate_set'), self.tf_record_validation_path, labels)
        self.create_tfrecord(os.path.join(self.data_dir, 'test_set'), self.tf_record_test_path, labels)

    def _train_validation_generator(self):
        ds_train_safe = self.get_tf_record(self.tf_record_training_safe_path)
        ds_train_undesired_content = self.get_tf_record(self.tf_record_training_undesired_content_path)

        t_s_s = self.get_data_size(ds_train_safe)
        t_s_n = self.get_data_size(ds_train_undesired_content)

        ds_train_safe = ds_train_safe.cache()
        ds_train_undesired_content = ds_train_undesired_content.cache()

        if t_s_s > t_s_n:
            ds_train_undesired_content = ds_train_undesired_content.repeat()
        elif t_s_s < t_s_n:
            ds_train_safe = ds_train_safe.repeat()

        ds_train_safe = ds_train_safe.shuffle(t_s_s)
        ds_train_undesired_content = ds_train_undesired_content.shuffle(t_s_n)

        ds_train_safe = self.get_data(ds_train_safe)
        ds_train_undesired_content = self.get_data(ds_train_undesired_content)

        ds_validation = self.get_tf_record(self.tf_record_validation_path)
        v_s = self.get_data_size(ds_validation)

        ds_validation = ds_validation.cache().shuffle(v_s)

        ds_validation = self.get_data(ds_validation)

        return ds_train_safe, ds_train_undesired_content, ds_validation

    def get_training_data_pipeline(self):
        train_ds_safe, train_ds_undesired_content, val_ds = self._train_validation_generator()
        train_ds_safe = self._apply_augmentation_to_data_pipe(train_ds_safe)
        train_ds_undesired_content = self._apply_augmentation_to_data_pipe(train_ds_undesired_content)

        train_ds_safe = self._apply_resize_and_norm_to_data_pipe(train_ds_safe).batch(self.train_config['batch_size'] // 2)
        train_ds_undesired_content = self._apply_resize_and_norm_to_data_pipe(train_ds_undesired_content).batch(self.train_config['batch_size'] // 2)

        train_ds = tf.data.Dataset.zip((train_ds_safe, train_ds_undesired_content)).map(self.stack_two_sides, num_parallel_calls=self.AUTO)

        train_ds = train_ds.prefetch(self.AUTO)

        val_ds = self._apply_resize_and_norm_to_data_pipe(val_ds)

        val_ds = val_ds.batch(self.train_config['batch_size']).prefetch(self.AUTO)
        return train_ds, val_ds


if __name__ == '__main__':
    from PIL import Image
    ft = FineTuningBase(None,'data/undesired_content')
    a = ft.read_csv_label_data()
