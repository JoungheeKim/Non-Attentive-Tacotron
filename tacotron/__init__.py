from tacotron.dataset import TGTProcessor
from tacotron.model import NonAttentiveTacotron
from tacotron.vocgan_generator import Generator
import torch

## In here you must sign your model with name
process_name = {
    'tgt' : TGTProcessor,
}

def get_process(cfg):
    name = cfg.process_name
    return process_name[name]

model_name = {
    'non_attentive_tacotron' : NonAttentiveTacotron
}

def get_model(cfg):
    name = cfg.model_name
    return model_name[name]


def get_vocgan(generator_path):
    """
        If you use pre-trained VocGAN model from rishikksh20,
        There is specific hyperparams to build VocGAN.
		git repository: https://github.com/rishikksh20/VocGAN
    :return: VocGAN
    """
    generator = Generator(80, 4,
          ratios=[4, 4, 2, 2, 2, 2], mult=256,
          out_band=1)
    generator_checkpoint = torch.load(generator_path)
    generator.load_state_dict(generator_checkpoint['model_g'])
    return generator
