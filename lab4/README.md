# MLOps. Практическое задание №4

### Цель: продемонстрировать навыки использования утилиты dvc

### Решение: 
В рамках выполнения задания исполнены все основные операции с dvc. Создал и несколько раз модифицировал датасет. Научился версионировать данные в Google Disc.

### Коммиты

```
45f49b1 (HEAD -> feature/lab4, origin/feature/lab4) new: dataset update - Sex OneHotEncoder
ba51cda new: dataset update - Age replace null to mean
bff27c6 new: dataset modification (Pclass, Sex, Age)
6675057 new: create dataset
a3ebcee add requirements.txt
dee0fc8 update readme
907eee2 update readme
56a439f update datasets.dvc
25f6307 add onehot encoding preprocessing
5d1e5f8 add null preprocessing
4f44805 dataset modification (Pclass, Sex, Age)
5c1ed75 update datasets.dvc
c310d13 add preprocess_df
b33653d test commit
ef8f65d add data remote
c90679e add dataset creator
a430146 add datasets.dvc and dataset gitignore
70bf708 init dvc
```

### Переключение между версиями датасета
Для переключения между версиями датасета использовал команды:
```
git log --oneline
git checkout <commit_id>
dvc pull
```
Для возвращения к текущей актуальной версии датасета
```
git checkout feature/lab4
dvc pull
```

### Структура каталогов и файлов
- src - файлы python скриптов
    - dataset_creator.py
        - создание датасета
    - preprocess_df.py
        - модификация полей датасета ('Pclass', 'Sex', 'Age')
    - preprocess_df_null.py
        - обновление датасета (null -Age)
    - preprocess_df_one_hot.py
        - обновление датасета (OneHotEncoder)
- datasets - отслеживаемая директория с датасетами (для dvc)
- requirements.txt - зависимости
- README.md - описание
- datasets.dvc - хеш наблюдаемого объекта
- .gitignore - правила игнорирования отслеживания файлов в git

### Запуск: в директории lab4/

```
python3 dataset_creator.py
python3 preprocess_df.py
python3 preprocess_df_null.py
python3 preprocess_df_one_hot.py
```

### Ссылки
- [Облачное хранилище Google Drive](https://drive.google.com/drive/folders/1PgosGabWxSnEtH4mpu1G7GOcdx4i_pux)
- [файл DVC для отслеживания версий данных](datasets.dvc)

### Требования:
Python 3.9+
Библиотеки:
pandas
scikit-learn
catboost

### Лицензия
MIT License