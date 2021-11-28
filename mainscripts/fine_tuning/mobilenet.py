from mainscripts.fine_tuning.base import FineTuningBase, FineTuningBaseSep
import tensorflow as tf


class FTMobilenet(FineTuningBaseSep):
    def __init__(self, model, data_dir, backbone_strategy):
        FineTuningBaseSep.__init__(self,
                                model=model,
                                data_dir=data_dir,
                                backbone_strategy=backbone_strategy
                                )
        self.unlock_after = None
        self.get_train_config()
        self.backup_dir = 'models/mobilenet'
        self.unlock_after = 'block_13_expand'
        self.freeze_batch_norm_layers = False
        self.sample_weight_coef = 12.0

    def get_train_config(self):
        self.train_config = {
            'backup_cycle': 3,
            'epoch': 100,
            'batch_size': 32,
            'learning_rate': 0.005,
            'momentum': 0.9,
            'decay': 0.9,
            'label_smoothing': 0.15,
            'dropout_rate': 0.20,
            'opt_epsilon': 1.0
        }

    def initialize_training_elements(self):
        self.global_epoch = tf.Variable(tf.constant(0, dtype='int64'), trainable=False, name='global_epoch')
        self.best_f1_score = tf.Variable(tf.constant(0., dtype='float32'), trainable=False, name='best_f1_score')
        self.loss = self.get_loss_function()
        # self.optimizer = tf.optimizers.RMSprop(learning_rate=self.train_config['learning_rate'],
        #                                        epsilon=self.train_config['opt_epsilon'],
        #                                        momentum=self.train_config['momentum'],
        #                                        decay=self.train_config['decay'])

        self.optimizer = self.use_adabelief_optimizer()

    def use_adabelief_optimizer(self):
        from adabelief_tf import AdaBeliefOptimizer
        return AdaBeliefOptimizer(learning_rate=1e-5,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  weight_decay=1e-2,
                                  amsgrad=False,
                                  epsilon=1e-9,
                                  rectify=False)

    def _wrapped_resize_and_norm(self):

        def resize_and_norm_data(image,label):
            image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])
            #image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        return resize_and_norm_data

    def convert_model_for_finetune(self):
        from mainscripts.fine_tuning.mobilenet_layer_map import DCT

        mnet = tf.keras.applications.MobileNetV2(alpha=1.4,
                                                 input_shape=(None, None, 3),
                                                 include_top=False,
                                                 weights=None,
                                                 pooling='max')

        org_model = tf.keras.layers.Dropout(self.train_config['dropout_rate'] * 2.5)(mnet.output)

        org_model = tf.keras.layers.Dense(5)(org_model)
        org_model = tf.keras.layers.Softmax()(org_model)
        org_model = tf.keras.Model(mnet.input, org_model)

        weights = self.model.weights

        def get_weight(name):
            for x in weights:
                if DCT[x.name] == name:
                    return x.value(), x.name
            return None

        weights_org = org_model.weights
        for w in weights_org:
            w_i, w_i_name = get_weight(w.name)
            w.assign(w_i)
            print(w.name, '--->', w_i_name)

        self.model = org_model

        if self.backbone_strategy == 'full':
            self.model.trainable = True

        elif self.backbone_strategy == 'freeze':
            self.model.trainable = False
            unlock = False

            for layers in self.model.layers:
                if layers.name == self.unlock_after:
                    unlock = True

                if unlock:
                    if self.freeze_batch_norm_layers and isinstance(layers, tf.keras.layers.BatchNormalization):
                        layers.trainable = False
                    else:
                        layers.trainable = True

                else:
                    layers.trainable = False
        elif self.backbone_strategy == 'partial':
            self.model.trainable = False

            for layers in self.model.layers[-2:]:
                layers.trainable = True
        else:
            raise AttributeError('wrong type of backbone strategy')

        out = tf.keras.layers.Dense(1, activation='sigmoid')(self.model.layers[-2].output)
        self.model = tf.keras.Model(self.model.input, out)


if __name__ == "__main__":
    from core.finetuned_mobilenet import UndesiredContentMobileNet

    FTMobilenet(UndesiredContentMobileNet().undesired_content_model, 'data/train_frames', backbone_strategy='freeze').training()