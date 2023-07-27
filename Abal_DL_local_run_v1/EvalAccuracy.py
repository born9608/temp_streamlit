import tensorflow as tf

class EvalAccuracy(tf.keras.metrics.Metric):
    """
    사용자 정의 평가지표 : Accuracy
    """
    def __init__(self, name="accuracy", **kwargs):
        super(EvalAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name=name, initializer="zeros")

    def update_state(self, y_true, y_predict, sample_weight=None):
        value = tf.abs((y_predict - y_true) / y_true)
        self.correct.assign(tf.reduce_mean(value))

    def result(self):
        return 1 - self.correct

    def reset_state(self):
        self.correct.assign(0.)

    def get_config(self):
        config = super(EvalAccuracy, self).get_config()
        # Add any custom configurations specific to your class
        return config

    @classmethod
    def from_config(cls, config):
        # Create an instance of your class from the config dictionary
        return cls(**config)