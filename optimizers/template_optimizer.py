from tensorflow.python.training.optimizer import Optimizer


class TemplateOptimizer(Optimizer):
    """Optimizer template
    """

    def __init__(self, lr, use_locking=False, name='TemplateOpt'):
        """Construct a new optimizer

        :param lr: A Tensor. The learning rate.
        :param use_locking: If True use locks for update operation.
        :param name: Optional name for the operations. Defaults to "TemplateOpt".
        """

        super(TemplateOptimizer, self).__init__(use_locking, name)
        self.lr = lr

    def _prepare(self):
        pass

    def _apply_dense(self, grad, var):
        pass

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass
