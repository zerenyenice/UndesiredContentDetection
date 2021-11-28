from mainscripts.fine_tuning.base import FineTuningBase, FineTuningBaseSep
import tensorflow as tf
from tensorflow_addons.layers import Maxout


class FTEfficientNet(FineTuningBaseSep):
    def __init__(self, model, data_dir, backbone_strategy):
        FineTuningBaseSep.__init__(self,
                                model=model,
                                data_dir=data_dir,
                                image_size=(296, 224),
                                backbone_strategy=backbone_strategy
                                )
        self.unlock_after = None
        self.get_train_config()
        self.backup_dir = 'models/efficientnet'
        self.unlock_after = 'block6a_expand_conv'
        self.freeze_batch_norm_layers = True
        self.sample_weight_coef = -8.0

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

        self.optimizer = self.use_adabelief_optimizer()

    def _wrapped_resize_and_norm(self):

        def resize_and_norm_data(image,label):
            image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])
            #image = tf.cast(image, tf.float32) / 255.0
            return image, label

        return resize_and_norm_data

    def use_adabelief_optimizer(self):
        from adabelief_tf import AdaBeliefOptimizer
        return AdaBeliefOptimizer(learning_rate=1e-5,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  weight_decay=1e-2,
                                  amsgrad=False,
                                  epsilon=1e-9,
                                  rectify=False)

    def convert_model_for_finetune(self):

        fnet = tf.keras.applications.EfficientNetB3(
                                                 input_shape=(None, None, 3),
                                                 include_top=False,
                                                 weights='imagenet',
                                                 pooling='avg')

        org_model = tf.keras.layers.Dropout(self.train_config['dropout_rate'] * 2.5)(fnet.output)
        #org_model = Maxout(512)(fnet.output)
        org_model = tf.keras.layers.Dense(128, activation='relu',
                                          kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                          kernel_regularizer=tf.keras.regularizers.l2(.00005))(org_model)
        org_model = tf.keras.layers.Dropout(self.train_config['dropout_rate'] * 2)(org_model)

        org_model = tf.keras.layers.Dense(64, activation='relu',
                                          kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                          kernel_regularizer=tf.keras.regularizers.l2(.00005))(org_model)
        org_model = tf.keras.layers.Dropout(self.train_config['dropout_rate'])(org_model)

        #org_model = tf.keras.layers.Dense(5, kernel_initializer="glorot_uniform")(org_model)
        org_model = tf.keras.Model(fnet.input, org_model)


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

            for layers in self.model.layers[-5:]:
                layers.trainable = True
        else:
            raise AttributeError('wrong type of backbone strategy')

        out = tf.keras.layers.Dense(1, activation='sigmoid')(self.model.layers[-1].output)
        self.model = tf.keras.Model(self.model.input, out)


if __name__ == "__main__":
    from core.finetuned_efficientnet import UndesiredContentMobileNet

    FTEfficientNet(UndesiredContentMobileNet().undesired_content_model, 'data/train_frames', backbone_strategy='freeze').training()