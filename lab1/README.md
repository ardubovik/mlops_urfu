# MLOps. Практическое задание №1

### Цель: создать простейший конвейер для автоматизации работы с моделью машинного обучения.

### Решение:
    Создан простейший конвейер для автоматизации работы с моделью машинного обучения. 
    Отдельные этапы конвейера машинного обучения описываются в разных python–скриптах, 
    которые соединяются и запускаются с помощью bash-скрипта.

### Структура каталогов и файлов
- src - файлы генерации, обработки данных, создания модели и тестирования данных
    - data_creation.py
        - функция генерации данных для обучения и тестирования
    - model_preprocessing.py
        - предобработка (стандартизация) данных
    - model_preparation.py
        - создание и обучение модели
        - метрики на тренировочных данных
    - model_testing.py
        - тестирование модели
        - оценки метрик обученной модели
    - common.py
        - класс хранения данных
        - класс получения метрик
- data - сгенерированные данные, сохраненная модель
- scripts - скрипт работы с виртуальным окружением, последовательный запуск файлов
- README.md - описание
- requirements.txt - зависимости

### Запуск:

1. Скачать проект
2. Важно!!! Перейти в каталог `lab1/scripts/` (pipeline.sh необходимо запускать непосредственно из каталога scripts)
3. Установить права на скрипт: `chmod +x pipeline.sh`
4. Запустить скрипт: `./pipeline.sh`

### Пример работы:

```
virtual environment is not activate
create virtual environment
activating virtual environment
install requirements

Collecting numpy ....

starting data creation
data saved to ../data/train/train_data.csv
data saved to ../data/test/test_data.csv
starting model preprocessing
data saved to ../data/train/train_data_preprocessed.csv
data saved to ../data/test/test_data_preprocessed.csv
starting model preparation
   Accuracy  Precision    Recall  F1-score
0  0.558904   0.559322  0.543956  0.551532
model trained and saved
starting model testing
accuracy on testing data: 0.97
   Accuracy  Precision    Recall  F1-score
0  0.967123   0.977528  0.956044  0.966667
```