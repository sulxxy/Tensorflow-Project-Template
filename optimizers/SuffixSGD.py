import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training.optimizer import Optimizer


class SuffixSGD(Optimizer):
    """Optimizer that implements the SuffixSGD algorithm
    See [E. Hazan et. al., 2016](https://arxiv.org/abs/1503.03712)
    """

    def __init__(self, lr, center, delta, sigma, use_locking=False, name='SuffixSGD'):
        """Construct a new SuffixSGD optimizer

        :param lr: A Tensor. The learning rate.
        :param center: A Tensor. The center of the decision set.
        :param delta: A Tensor. The diameter of the decision set.
        :param sigma: A Tensor or floating point value. The niceness of the loss function.
        :param use_locking: If True use locks for update operation.
        :param name: Optional name for the operations. Defaults to "SuffixSGD".
        """

        super(SuffixSGD, self).__init__(use_locking, name)
        self._sigma = sigma
        self._lr_tensor = lr
        self._delta_tensor = delta
        self._center_tensor = center
        self._sigma_tensor = None

    def _prepare(self):
        sigma = self._call_if_callable(self._sigma)
        self._sigma_tensor = ops.convert_to_tensor(sigma, name='sigma')

    def _apply_dense(self, grad, var):
        var_update_op = state_ops.assign_sub(var, self._lr_tensor * grad)
        var_projection = self._delta_tensor * var / tf.norm(var) + self._center_tensor
        var_projection_op = state_ops.assign(var, var_projection)
        return control_flow_ops.group(*[var_update_op, var_projection_op])

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

    def _resource_apply_sparse(self, grad, handle, indices):
        return self._apply_sparse(grad, handle)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not implemented.")
