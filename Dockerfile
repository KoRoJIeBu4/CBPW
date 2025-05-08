FROM python:3.10-slim

# Устанавливаем зависимости для сборки и pip
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --upgrade pip

WORKDIR /app

# Копируем и устанавливаем зависимости проекта
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем необходимые компоненты для работы Voilà и поддержки Dash внутри Voila
# jupyter notebook + ipykernel нужны для запуска ядра Python
# jupyter-dash и voila-dash добавляют поддержку Dash-приложений в Voilà
RUN pip install --no-cache-dir \
    voila \
    notebook \
    ipykernel \
    jupyter-dash \
    voila-dash

# Копируем весь код приложения
COPY . .

# Открываем порт для приложения (Render использует его автоматически)
EXPOSE 7860

# Запуск Voilà на указанном ноутбуке
CMD ["voila", "vidgets.ipynb", "--port=7860", "--no-browser", "--Voila.ip=0.0.0.0"]
