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


### Решение соревнования

Прежде всего надо разделить данные на валидационную и тестовую. Запустите скрипт split_dataset.py, он разделит данные и создаст отдельные папки
с валидационной и тестовой выборками
```
python3 split_dataset.py -d ./data/traint.csv
```

Мы будем пробовать различные opensource сети для обработки изображений, такие как resnet, mobilenet, googlenet, etc. 
К каждой такой сети в конце нужно прикрепить голову для классификации. Определение классов вы можете найти в файле 
__model.py__. Вы можете добавлять ваши новые классы или меня форму классификационной головы, которая вынесена в отдельный 
класс. 

В начале мы соберем данные по тому, как различные модели справляются с задачей. Для этого запустим подготовленный скрипт
```
python3 run.py
```

Этот файл прогонит задачу по всем подготовленным нейронкам, сохранит графики обучения и валидации в папке __images__ и
сохраняет лучшие модели в дериктории __saved_models__. 

__ATTENTION:__ поскольку скрипт прогоняет эксперимент по ВСЕМ подготовленным моделям, это может занять большое количество 
времени. Обратите также внимание, что отладочная информация об эксперименте записывается в лог файл __logs/core.log__.

После того, как модели обучились и прошли первичную валидацию, вы можете провести вторичную валидацию, а именно, прогнать
наилучшие сохраненные модели на валидационной части датасета, отрисовать матрицы конволюции и посмотреть детальные
метрики по классификации. Для этого мы используем скрипт
```
python3 eval_model.py -m ./saved_models/best_model_resnet.pth
```
Для запуска в аргументах скрипта укажите путь к сохраненной модели, которую хотите протестировать. Скрипт выведет 
детальную статистику о классификации, а так же сохранит график с матрицей конволюции в дериктории __images__