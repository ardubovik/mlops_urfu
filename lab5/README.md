# MLOps. Практическое задание №5

### Цель: Применить средства автоматизации тестирования python для автоматического тестирования качества работы модели машинного обучения на различных датасетах.

### Решение: 
В рамках выполнения задания исполнены создал 3 датасета без шума и 1 с шумом, создал модель линейной регрессии, обучил её на 2м датасете и протестировал все датасеты на MSE, научился выявлять ошибки через тесты.

### Структура каталогов и файлов

- src/ - каталог содержит тесты и ноутбуки
  - mlops_lab5_run_from_def.ipynb - ноутбук для запуска через функцию - как есть
  - MLOps_lab5.ipynb - ноутбук для запуска через pytest
  - test_df_mse.py - файл, сохранённый через ноутбук MLOps_lab5 для запуска в pytest
- requirements.txt - зависимости
- README.md - описание

### Запуск тестов 
- Для запуска через pytest в коллабе запустить поочерёдно блоки MLOps_lab5.ipynb
- Для запуска с графиками через функцию - установить зависимости и запустить блоки mlops_lab5_run_from_def.ipynb или в интерпретаторе файл test_df_mse.py

### Требования:
Python 3.9+
Библиотеки:
pandas
numpy
scikit-learn
matplotlib

### Лицензия
MIT License