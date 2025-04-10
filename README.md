## Preparation

This is a work of the kaggle competiotion. We will use a kaggle API to download data and submission.

First of all you need to do some prerequests:
1/ Create a python virtual enviroment and check our that you have installed kaggle in this env
```
python3 -m venv ./.venv
source ./.venc/bin/activate
pip install kaggle
```
2/ verify, that you have an account on kaggle
3/ verify you have a kaggle token (you can create and download new token on you kaggle account in settings)
4/ verify your kaggle token is on you local computer in directory ~/.config/kaggle/kaggle.json
5/ verify your kaggle token has right access
```
sudo chmod 600 ~/.config/kaggle/kaggle.json
```
6/ verify, you take part in a competition you want to solve (go to your kaggle web page of competition to the data and push a button take part in competition)

after you have verified all of this aspects, you can use kaggle API to download a data for your competition
```
kaggle competitions download -c plant-pathology-2020-fgvc7
```

next you can unzip this file
```
unzip ./plant-pathology-2020-fgvc7.zip -p ./data
```


## Решение соревнования

Прежде всего надо разделить данные на валидационную и тестовую. Запустите скрипт split_dataset.py, он разделит данные и создаст отдельные папки
с валидационной и тестовой выборками
```
python3 ./src/split_dataset.py -d ./data/traint.csv
```

Мы будем пробовать различные opensource сети для обработки изображений, такие как resnet, mobilenet, googlenet, etc. 
К каждой такой сети в конце нужно прикрепить голову для классификации. Определение классов вы можете найти в файле 
__model.py__. Вы можете добавлять ваши новые классы или меня форму классификационной головы, которая вынесена в отдельный 
класс. 

В начале мы соберем данные по тому, как различные модели справляются с задачей. Для этого запустим подготовленный скрипт
```
mkdir images
mkdir saved_models
python3 ./src/run.py
```

Этот файл прогонит задачу по всем подготовленным нейронкам, сохранит графики обучения и валидации в папке __images__ и
сохраняет лучшие модели в дериктории __saved_models__. 

__ATTENTION:__ поскольку скрипт прогоняет эксперимент по ВСЕМ подготовленным моделям, это может занять большое количество 
времени. Обратите также внимание, что отладочная информация об эксперименте записывается в лог файл __logs/core.log__.

После того, как модели обучились и прошли первичную валидацию, вы можете провести вторичную валидацию, а именно, прогнать
наилучшие сохраненные модели на валидационной части датасета, отрисовать матрицы конволюции, посмотреть детальные
метрики по классификации и отранжировать исследованные модели. Для этого мы используем скрипт
```
./scripts/test_eval_models.sh
```
Для запуска в аргументах скрипта укажите путь к сохраненной модели, которую хотите протестировать. 
__ATTENTION:__ обратите внимание, что архитектура сети в файле __model.py__ должна полностью совпадать с архитектурой сети, сохраненной в файле .pth,
в противном случае программа не сможет корректно загрузить веса модели.
Скрипт выведет детальную статистику о классификации, сохранит график с матрицей конволюции в дериктории __images__ и запишет в файл
model_evaluation_result.csv имена моделей и их финальные метрики accuracy. После этого в этом же скрипте отработает логика по 
автоматическому ранжированию моделей и в консоль выведется список top моделей по итогу прогона.

Наконец, после того, как модели обучены и прошли валидацию, можно запустить скрипт, который генерирует предсказания для тестовой выборки.
Вы можете запустить скрипт для какой-то конкретной модели
```
python3 ./src/predict.py -m ./saved_models/head_1/best_model_resnet.pth
```
Или сделать предсказание на всех обученных моделях и потом исследовать и сравнивать их.
```
./srcipts/make_models_prediction.sh
```
Во втором случае файлы с предсказаниями для каждой модели сохранятся в каталог submissions.

## Результаты

## Отправление результата на kaggle

```
kaggle competitions submit -c plant-pathology-2020-fgvc7 -f submission.csv -m "Message"
```

## For developers

```
python3 ./third-party/compare.py -d ./submissions/ -e ./tmp/submission.csv
```