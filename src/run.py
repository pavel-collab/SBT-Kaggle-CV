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
                             train_dataset,
                             val_dataset,
                            #  train_loader,
                            #  val_loader,
                             class_weights
                             )

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.classification_head import ClassificationHead4
from models.model import CustomConvNeXtTiny

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume', action='store_true', help='resume mode')
args = parser.parse_args()

define_random_seed()
device = get_device()
    
# if args.resume:
#     logfile_path = Path(DEFAULT_LOG_PATH)
#     assert(logfile_path.exists())

#     last_model_name, last_head_name = last_model_settings(logfile_path.absolute())
#     logger.info(f"Resume model training from {last_model_name} and {last_head_name}")
# else:
#     last_model_name, last_head_name = list(models.keys())[0], list(classification_heads.keys())[0]

# try:
#     with profiler.profile(
#     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
#     ) as prof:
#         for model_name, model_class in dropwhile(lambda item: item[0] != last_model_name, models.items()):
#             for class_head_name, cl_head in dropwhile(lambda item: item[0] != last_head_name, classification_heads.items()):
#                 logger.info(f"Start to train model {model_name} with classification head {class_head_name}")
#                 model = model_class(classification_head=cl_head, n_classes=n_classes)
#                 try:
#                     model_train_result = train_model(model,
#                                                     model_name,
#                                                     device,
#                                                     num_epochs,
#                                                     learning_rate,
#                                                     train_loader,
#                                                     val_loader,
#                                                     classification_head_name=class_head_name,
#                                                     class_weights=class_weights
#                                                     )
#                     plot_train_proces(num_epochs,
#                                     model_train_result.train_losses,
#                                     model_train_result.val_losses,
#                                     model_train_result.train_accuracies,
#                                     model_train_result.val_accuracies,
#                                     model_name,
#                                     classification_head_name=class_head_name)
#                 except Exception as ex:
#                     logger.error(f"During training model {model_name} was caught exception {ex}")
#                     continue
# except KeyboardInterrupt:
#     print("The program was interrupted by key signal")

def main():
    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=42)
    # parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    # parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    # model_dir = argv.model_dir
    # model_filename = argv.model_filename
    resume = argv.resume

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    # torch.distributed.init_process_group(backend="gloo")

    model = CustomConvNeXtTiny(n_classes=n_classes, classification_head=ClassificationHead4, pretrained=True)
    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load('last_model.pth', map_location=map_location)) #! now we temporary hardcode save filepath here
    
    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Определим функцию потерь и оптимизатор
    if class_weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0

    # model.to(device)

    for epoch in range(num_epochs):
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

        ddp_model.train()
        running_loss = 0.0 # loss в рамках 1 прохода по датасету (одной эпохи)
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # зануляем градиенты перед обработкой очередного батча
            outputs = ddp_model(images) # получаем предсказания модели

            loss = criterion(outputs, labels) # получаем выход функции потерь
            loss.backward() # прогоняем градиенты обратно по графу вычиялений от хвоста сети к голове
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #! добавить клипинк для предотвращения взрыва градиентов
            
            optimizer.step() # делаем шаг градиентного спуска (обновляем веса)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader) # средняя ошибка за один проход по данным (за 1 эпоху)
        train_accuracy = correct / total
        # сохраняем данные по эпохе
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        if local_rank == 0:
            # Валидация модели
            ddp_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = ddp_model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                        f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
            scheduler.step(val_loss) # добавляем уменьшене learning rate
        
            # Сохранение лучшей модели на основе валидационной точности
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(ddp_model.state_dict(), f"best_model.pth")
                print('Saved best model!')
            
            torch.save(ddp_model.state_dict(), f"last_model.pth")

    logger.info(f"Training and validation complete!")


if __name__ == '__main__':
    main()
