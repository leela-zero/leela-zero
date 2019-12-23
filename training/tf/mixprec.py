import tensorflow as tf


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision."""
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer,
                      regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        cast_name = name + '/fp16_cast'
        try:
            cast_variable = tf.get_default_graph().get_tensor_by_name(
                cast_name + ':0')
        except KeyError:
            cast_variable = tf.cast(variable, dtype, name=cast_name)
        cast_variable._ref = variable._ref
        variable = cast_variable
    return variable


class LossScalingOptimizer(tf.train.Optimizer):
    """An optimizer that scales loss and un-scales gradients."""

    def __init__(self, optimizer,
                 scale=None,
                 name="LossScalingOptimizer",
                 use_locking=False):
        super(LossScalingOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._scale = float(scale) if scale is not None else 1.0

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        if self._scale != 1.0:
            loss = tf.scalar_mul(self._scale, loss)
        gradvar = self._optimizer.compute_gradients(loss, var_list, *args, **kwargs)
        gradvar = [(tf.scalar_mul(1. / self._scale, g), v) for g, v in gradvar]
        return gradvar

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)
