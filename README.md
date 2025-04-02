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