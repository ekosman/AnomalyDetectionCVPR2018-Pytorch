from os import path
from pytorch_wrapper.training_callbacks import AbstractCallback


class TensorboardCallback(AbstractCallback):
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    def post_loss_calculation(self, training_context):
        if training_context['current_loss'] is not None:
            self.tb_writer.add_scalar('Train loss',
                                      training_context['current_loss'].item(),
                                      training_context['_current_epoch'])


class SaveCallback(AbstractCallback):
    def __init__(self, target_dir, model_name, save_every):
        self.target_dir = target_dir
        self.model_name = model_name
        self.save_every = save_every

    def on_epoch_end(self, training_context):
        epoch = training_context['_current_epoch']
        if epoch % self.save_every != 0:
            training_context['system'].save_model_state(path.join(self.target_dir, f'{self.model_name}_{epoch}.weights'))