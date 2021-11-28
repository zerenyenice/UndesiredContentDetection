from mainscripts.fine_tuning.base import FineTuningBase
import tensorflow as tf


class FTInception(FineTuningBase):
    def __init__(self, model, data_dir, backbone_strategy='freeze'):

        FineTuningBase.__init__(self,
                                model=model,
                                data_dir=data_dir,
                                backbone_strategy=backbone_strategy
                                )
        self.get_train_config()
        self.backup_dir = 'models/inception'
        self.unlock_after = 'conv2d_56'
        self.freeze_batch_norm_layers = False

    def get_train_config(self):
        self.train_config = {
            'backup_cycle': 3,
            'epoch': 100,
            'batch_size': 64,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'decay': 0.9,
            'label_smoothing': 0.05,
            'dropout_rate': 0.10,
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

        self.optimizer = tf.optimizers.Adam(lr=self.train_config['learning_rate'], amsgrad=True)

    def convert_model_for_finetune(self):
        base_model = tf.keras.applications.InceptionV3(
            include_top = False,
            input_shape=(None, None, 3),
            weights=None,
            pooling='avg'
        )

        org_model = tf.keras.layers.Dense(256, activation='relu',
                                          kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                          kernel_regularizer=tf.keras.regularizers.l2(.0005))(base_model.output)
        org_model = tf.keras.layers.Dropout(self.train_config['dropout_rate'] * 2)(org_model)

        org_model = tf.keras.layers.Dense(128, activation='relu',
                                          kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                          kernel_regularizer=tf.keras.regularizers.l2(.0005))(org_model)
        org_model = tf.keras.layers.Dropout(self.train_config['dropout_rate'])(org_model)

        org_model = tf.keras.layers.Dense(5, kernel_initializer="glorot_uniform")(org_model)

        org_model = tf.keras.Model(base_model.input, org_model)
        org_model.set_weights(self.model.get_weights())

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
    from core.finetuned_inception import UndesiredContentInception

    FTInception(UndesiredContentInception().undesired_content_model, 'data/train_frames', backbone_strategy='finetune').convert_model_for_finetune()