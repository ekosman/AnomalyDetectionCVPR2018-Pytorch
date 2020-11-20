from pytorch_wrapper.training_callbacks import AbstractCallback


class TensorboardCallback(AbstractCallback):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    def post_loss_calculation(self, training_context):
        print(training_context)
        if training_context['current_loss'] is not None:
            self.tb_writer.add_scalars('Train loss',
                                       training_context['current_loss'],
                                       training_context['_current_epoch'])
