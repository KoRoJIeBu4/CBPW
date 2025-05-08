FROM python:3.10-slim

# Устанавливаем зависимости для сборки и pip
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --upgrade pip

WORKDIR /app

# Копируем и устанавливаем зависимости проекта
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir voila
# Устанавливаем необходимые компоненты для работы Voilà
# jupyter notebook + ipykernel нужны для запуска ядра Python
RUN pip install --no-cache-dir \
    voila \
    notebook \
    ipykernel

# Копируем весь код приложения
COPY . .

# Открываем порт для приложения (Render использует его автоматически)
EXPOSE 7860

# Запуск Voilà на указанном ноутбуке
CMD ["voila", "vidgets.ipynb", "--port=7860", "--no-browser", "--Voila.ip=0.0.0.0"]