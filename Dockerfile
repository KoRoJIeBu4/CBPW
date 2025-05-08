# Берём Python 3.9 Slim (тогда pip найдёт подходящие колёса для voila)
FROM python:3.9-slim

# Заливаем системные сборочные инструменты и обновляем pip
RUN apt-get update && apt-get install -y \
    build-essential \
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
CMD ["voila", "vidgets.ipynb", "--port=7860", "--no-browser", "--Voila.ip=0.0.0.0"]