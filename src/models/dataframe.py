import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):
    def __init__(self, csv_file, images_dir, classes: list, transform=None, is_test=False):
        """
        Инициализация кастомного датафрейма.

        :param csv_file: Путь к CSV файлу с информацией о метках классов.
        :param images_dir: Путь к папке с изображениями.
        :param transform: Объект типа torchvision.transforms для преобразования изображений.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.classes = classes
        self.is_test = is_test

    def __len__(self):
        """Возвращает количество изображений в наборе данных."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Возвращает изображение и метку класса по индексу.

        :param idx: Индекс изображения.
        :return: Кортеж (изображение, метка класса).
        """
        # Получаем данные из датафрейма
        img_name = os.path.join(self.images_dir, self.dataframe.iloc[idx, 0]) + ".jpg"  # Предполагается, что имя изображения в первом столбце
        
        if not self.is_test: 
            label = int(self.dataframe.iloc[idx,1:].values.argmax()) # Предполагается, что метки закодированы one hot с первой позиции
        
        # Загружаем изображение
        # image = Image.open(img_name).convert('RGB')
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Применяем преобразования, если они есть
        if self.transform:
            image = self.transform(image=img)["image"]

        if not self.is_test:
            return (image, label)
        else:
            return image
        