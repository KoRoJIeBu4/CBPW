# Берём Python 3.9 Slim (тогда pip найдёт подходящие колёса для voila)
FROM python:3.9-slim

# Устанавливаем зависимости для сборки, pip и HDF5 (для PyTables)
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    && pip install --upgrade pip

WORKDIR /app

# Копируем зависимости и ставим их
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем Voilà и всё, что нужно для Dash в Voilà
RUN pip install --no-cache-dir \
    voila \
    notebook \
    ipykernel \
    jupyter-dash

# Копируем код
COPY . .

# Открываем порт — на Render он будет доступен на $PORT
EXPOSE 7860

# Запускаем Voilà на вашем ноутбуке
CMD ["voila", "vidgets.ipynb", "--port=7860", "--no-browser", "--show_tracebacks=True", "--Voila.ip=0.0.0.0"]