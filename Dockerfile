FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8866

CMD ["voila", "vidgets.ipynb", "--port=8866", "--no-browser", "--Voila.ip=0.0.0.0"]