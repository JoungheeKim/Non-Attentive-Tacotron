import logging
import os
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import random
import hydra
from datetime import datetime


def get_abspath(path):
    if not os.path.isabs(path):
        try:
            path = os.path.join(hydra.utils.get_original_cwd(), path)
        except:
            print("can't use hydra lib. the path will be [{}]".format(path))
    return path



def set_seed(seed):
    """
        Freeze all random seed to train model with consistency
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)



def reset_logging():
    """
        logging settings
    """
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


class ResultWriter:
    def __init__(self, directory):

        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, base_cfg, **results):
        now = datetime.now()
        date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update(dict(base_cfg))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()
        logging.info("save experiment info [{}]".format(self.dir))

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None

