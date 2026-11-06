import swanlab
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

from utils.paths import storage_path


class SummaryWriter(TorchSummaryWriter):
    def __init__(self, log_dir=None, *args, **kwargs):
        if log_dir is not None:
            log_dir = storage_path(log_dir)
        super().__init__(log_dir=log_dir, *args, **kwargs)

class Logger:
    def __init__(self, log_swanlab=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_swanlab = log_swanlab
    def add_scalar(self, tag, scalar_value, step):
        if self.log_swanlab:
            swanlab.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step):
        """Log one HWC uint8 image (what the encoder is fed, after augmentation)."""
        if self.log_swanlab:
            swanlab.log({tag: swanlab.Image(image)}, step=step)
        # TensorBoard wants CHW by default; say HWC instead of transposing.
        self.writer.add_image(tag, image, step, dataformats="HWC")

    def close(self):
        self.writer.close()
