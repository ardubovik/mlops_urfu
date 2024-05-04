FROM python:3.9

USER root

# установка зависимостей
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# копирование файлов проекта
COPY . /app
WORKDIR /app/src

# все процессы получения и подготовки данных
RUN python3 main.py

# запуск приложения streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8081", "--server.address=0.0.0.0"]
