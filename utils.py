import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
import logging

import logger_config

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

# Получение метрик качества для текущих весов модели
def evaluate_model(model, dataloader, device):
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

# Функция для построения графика матрицы ошибок
def plot_confusion_matrix(cm, classes):
    with plt.style.context('default'):  
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
        
def train_model(model, 
                model_name: str, 
                device, 
                num_epochs: int, 
                learning_rate: float, 
                train_loader, 
                val_loader) -> TrainModelResult:
    # Определим функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # зануляем градиенты перед обработкой очередного батча
            outputs = model(images) # получаем предсказания модели

            loss = criterion(outputs, labels) # получаем выход функции потерь
            loss.backward() # прогоняем градиенты обратно по графу вычиялений от хвоста сети к голове
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
    
        # Сохранение лучшей модели на основе валидационной точности
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"{MODEL_SAVE_DIR_PATH}/best_model_{model_name}.pth")
            logger.info('Saved best model!')
        
        # Сохранение последней актуальной модели
        torch.save(model.state_dict(), f"{MODEL_SAVE_DIR_PATH}/last_model_{model_name}.pth")

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
                      model_name: str):
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

    plt.savefig(f"{IMAGE_DIR_PATH}/{model_name}.jpg")