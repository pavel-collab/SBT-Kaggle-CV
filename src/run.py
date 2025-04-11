import torch 
import numpy as np
import argparse
from itertools import dropwhile
from pathlib import Path
import logging

from utils.utils import (train_model, 
                         plot_train_proces, 
                         last_model_settings, 
                         define_random_seed,
                         get_device)
import torch.profiler as profiler

import utils.logger_config as logger_config
from utils.logger_config import DEFAULT_LOG_PATH

from utils.constants import (models,
                             classification_heads,
                             num_epochs,
                             n_classes,
                             learning_rate,
                             train_loader,
                             val_loader,
                             class_weights)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume', action='store_true', help='resume mode')
args = parser.parse_args()

define_random_seed()
device = get_device()
    
if args.resume:
    logfile_path = Path(DEFAULT_LOG_PATH)
    assert(logfile_path.exists())

    last_model_name, last_head_name = last_model_settings(logfile_path.absolute())
    logger.info(f"Resume model training from {last_model_name} and {last_head_name}")
else:
    last_model_name, last_head_name = list(models.keys())[0], list(classification_heads.keys())[0]

try:
    with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
    ) as prof:
        for model_name, model_class in dropwhile(lambda item: item[0] != last_model_name, models.items()):
            for class_head_name, cl_head in dropwhile(lambda item: item[0] != last_head_name, classification_heads.items()):
                logger.info(f"Start to train model {model_name} with classification head {class_head_name}")
                model = model_class(classification_head=cl_head, n_classes=n_classes)
                try:
                    model_train_result = train_model(model,
                                                    model_name,
                                                    device,
                                                    num_epochs,
                                                    learning_rate,
                                                    train_loader,
                                                    val_loader,
                                                    classification_head_name=class_head_name,
                                                    class_weights=class_weights
                                                    )
                    plot_train_proces(num_epochs,
                                    model_train_result.train_losses,
                                    model_train_result.val_losses,
                                    model_train_result.train_accuracies,
                                    model_train_result.val_accuracies,
                                    model_name,
                                    classification_head_name=class_head_name)
                except Exception as ex:
                    logger.error(f"During training model {model_name} was caught exception {ex}")
                    continue
except KeyboardInterrupt:
    print("The program was interrupted by key signal")