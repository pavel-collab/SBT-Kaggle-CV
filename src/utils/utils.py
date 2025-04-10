import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
import logging
import os
from pathlib import Path
import re
import torch.nn.functional as F

import utils.logger_config

logger = logging.getLogger(__name__)

MODEL_SAVE_DIR_PATH = './saved_models'
IMAGE_DIR_PATH = "./images"

class TrainModelResult():
    def __init__(self, 
                 train_losses: list, 
                 val_losses: list, 
                 train_accuracies: list, 
                 val_accuracies: list):
        self.train_losses = train_losses 
        self.val_losses = val_losses 
        self.train_accuracies = train_accuracies 
        self.val_accuracies = val_accuracies

def define_random_seed(seed=20):
    """
    Set random seeds for reproducibility across different libraries.

    This function sets consistent random seeds for PyTorch CPU operations,
    CUDA operations, and NumPy to ensure reproducible results across runs.
    It also configures CUDA backend settings for deterministic behavior.

    Args:
        seed (int, optional): The random seed value to use for all random number
                             generators. Defaults to 20.
    """
    # фиксируем рандомный сид
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
def get_device():
    # детектируем девайс
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def evaluate_model(model, dataloader, device):
    """
    Evaluates a PyTorch model's performance on a given dataset.

    This function runs the model in evaluation mode on the provided dataloader,
    collects predictions, and computes various performance metrics including
    confusion matrix, classification report, accuracy, and weighted F1 score.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader containing the evaluation dataset.
        device: The computation device (CPU or GPU) where the model and data should be loaded.

    Returns:
        tuple: A tuple containing:
            - cm (numpy.ndarray): Confusion matrix of model predictions.
            - report (str): Classification report with precision, recall, and F1 score for each class.
            - accuracy (float): Overall accuracy of the model on the dataset.
            - weighted_f1 (float): Weighted F1 score across all classes.
    """
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    # Вычисление взвешенной F1-меры для текущей модели
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    return cm, report, accuracy, weighted_f1

def plot_confusion_matrix(cm, classes, model_name=None, save_file_path=None):
    """
    Plots a confusion matrix for visualizing classification performance.

    This function takes the confusion matrix and class labels to create a heatmap
    visualization. It also allows saving the plot to a file or returning it
    without saving.

    Args:
        cm (numpy.ndarray): The confusion matrix array.
        classes (list): List of class names used in the model.
        model_name (string, optional): Name of the model for naming purposes. If None,
                                        does not set a title. Defaults to None.
        save_file_path (str, optional): Path where the plot should be saved. If None,
                                         the plot is displayed but not saved. Defaults to None.

    Returns:
        str: The filename or None if no saving occurs.

    Raises:
        AssertionError: If model_name is provided but save_file_path is not set.
    """
    with plt.style.context('default'):  
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        if model_name:
            assert save_file_path is not None
            plt.title(f"Confusion Matrix for {model_name}")
        else:
            plt.title("Confusion Matrix")
        
        if save_file_path is None:
            plt.show()
        else:
            # Verify that model_name exists before saving
            assert model_name, "model_name must be provided when save_file_path is not None"
            plt.savefig(f"{save_file_path}/confusion_matrix_{model_name}.jpg")
            return f"{save_file_path}/confusion_matrix_{model_name}.jpg"
        
def train_model(model, 
                model_name: str, 
                device, 
                num_epochs: int, 
                learning_rate: float, 
                train_loader, 
                val_loader,
                classification_head_name=None,
                class_weights=None) -> TrainModelResult:
    # Определим функцию потерь и оптимизатор
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0 # loss в рамках 1 прохода по датасету (одной эпохи)
        correct = 0
        total = 0
        
        # динамическое сглаживание
        smoothing = 0.2 * (1 - epoch / num_epochs)

        if class_weights is None:
            criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=smoothing)

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # зануляем градиенты перед обработкой очередного батча
            logits = model(images) # получаем предсказания модели

            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, labels)

            loss.backward() # прогоняем градиенты обратно по графу вычиялений от хвоста сети к голове
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #! добавить клипинк для предотвращения взрыва градиентов
            
            optimizer.step() # делаем шаг градиентного спуска (обновляем веса)
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader) # средняя ошибка за один проход по данным (за 1 эпоху)
        train_accuracy = correct / total
        # сохраняем данные по эпохе
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Валидация модели
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
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
            if classification_head_name is None:
                torch.save(model.state_dict(), f"{MODEL_SAVE_DIR_PATH}/best_model_{model_name}.pth")
            else:
                path_to_save = Path(f"{MODEL_SAVE_DIR_PATH}/{classification_head_name}")
                if not path_to_save.exists():
                    os.mkdir(path_to_save.absolute())
                torch.save(model.state_dict(), f"{path_to_save.absolute()}/best_model_{model_name}.pth")
            logger.info('Saved best model!')
        
        # Сохранение последней актуальной модели
        if classification_head_name is None:
            torch.save(model.state_dict(), f"{MODEL_SAVE_DIR_PATH}/last_model_{model_name}.pth")
        else:
            path_to_save = Path(f"{MODEL_SAVE_DIR_PATH}/{classification_head_name}")
            if not path_to_save.exists():
                os.mkdir(path_to_save.absolute())
            torch.save(model.state_dict(), f"{path_to_save.absolute()}/last_model_{model_name}.pth")

    logger.info(f"Training and validation complete!")
    
    return TrainModelResult(train_losses,
                            val_losses,
                            train_accuracies,
                            val_accuracies)

def plot_train_proces(num_epochs: int,
                      train_losses,
                      val_losses,
                      train_accuracies,
                      val_accuracies,
                      model_name: str,
                      classification_head_name=None):
    # Построим графики
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss vs. Epoch ({model_name})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy vs. Epoch ({model_name})")
    plt.legend()

    if classification_head_name is None:
        plt.savefig(f"{IMAGE_DIR_PATH}/{model_name}.jpg")
    else:
        plt.savefig(f"{IMAGE_DIR_PATH}/{model_name}_{classification_head_name}.jpg")

def last_model_settings(log_file_path: str):
    pattern = r"[\d,-_:\s*]+ - INFO - Start to train model ([\d,\w,_]+) with classification head ([\d,\w,_]+)"
    
    last_model_name = None
    last_class_head_name = None

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                last_model_name = match.group(1)  # Извлекаем model_name
                last_class_head_name = match.group(2)  # Извлекаем class_head_name

    if last_model_name and last_class_head_name:
        return last_model_name, last_class_head_name
    else:
        raise RuntimeError("logfile exists, but there are no records about last models training")
    
def extract_model_name(model_file_name: str):
    if 'best_model' in model_file_name:
        return model_file_name.removeprefix("best_model_").removesuffix(".pth")
    elif 'last_model' in model_file_name:
        return model_file_name.removeprefix("last_model_").removesuffix(".pth")
    else:
        return None