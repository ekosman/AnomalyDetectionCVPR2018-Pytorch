from pytorch_wrapper.training_callbacks import AbstractCallback


class TensorboardCallback(AbstractCallback):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    def on_epoch_end(self, training_context):
        print(training_context)
        self.tb_writer.add_scalars('Train loss',
                                   training_context.current_loss,
                                   training_context._current_epoch)
