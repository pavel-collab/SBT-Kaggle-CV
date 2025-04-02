import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, images_dir, classes: list, transform=None):
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
        label = int(self.dataframe.iloc[idx,1:].values.argmax()) # Предполагается, что метки закодированы one hot с первой позиции
        
        # Загружаем изображение
        image = Image.open(img_name).convert('RGB')

        # Применяем преобразования, если они есть
        if self.transform:
            image = self.transform(image)

        return (image, label)

'''
Пример использования

# Параметры
csv_file = 'path/to/your/train.csv'
images_dir = 'path/to/your/train/images'
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Измените размер изображений
    transforms.ToTensor(),            # Преобразование в тензор
])

# Создание кастомного датафрейма
dataset = CustomDataset(csv_file, images_dir, transform)

# Создание DataLoader
from torch.utils.data import DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Итерация по DataLoader
for batch_idx, (images, labels) in enumerate(data_loader):
    print(f'Batch {batch_idx + 1}:')
    print(f'  Images batch shape: {images.shape}')
    print(f'  Labels: {labels}')
'''