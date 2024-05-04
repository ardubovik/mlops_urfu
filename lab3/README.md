# MLOps. Практическое задание №3

### Цель: развернуть микросервис в контейнере docker.

### Решение: реализован пайплайн обучения модели и приложение на `streamlit`. Запуск происходит через `docker-compose`, пайплайн описан в `Dockerfile`.

### Структура каталогов и файлов
- src - файлы общего файла main, приложения app и общих обработчиков
    - app.py
        - реализация приложения на streamlit
    - model_preprocessor.py
        - предобработка (стандартизация) данных
    - model_preparation.py
        - создание и обучение модели
    - logger.py
        - логирование
    - main.py
        - общий файл имплементаций
    - common.py
        - класс хранения данных
        - класс получения метрик
        - класс DataCreator (разбиение на тренировочную и тестовую)
    - .streamlit/config.toml - конфигурация настроек для streamlit
- docker-compose.yml - конфигурация запуска приложения на основе файла `Dockerfile`
- requirements.txt - зависимости
- README.md - описание
- Dockerfile - описание пайплайна установки и запуска приложения

### Запуск: в директории lab3/

```docker-compose up -d```

### Результат работы:
1. сначала отработает загрузка, предподготовка и сохранения данных, далее запуск приложения по адресу: `http://localhost:8081/`
2. перейдя в браузере по адресу `http://localhost:8081/` можно настроить значения для предсказания
3. нажать кнопку `Predict` и получить результат определения вида.

### Требования:
Python 3.9+
Библиотеки:
numpy
pandas
scikit-learn
joblib
streamlit

### Лицензия
MIT License