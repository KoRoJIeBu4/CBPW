FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --upgrade pip

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN pip install voila

COPY . .

EXPOSE 7860

CMD ["voila", "vidgets.ipynb", "--port=7860", "--no-browser", "--Voila.configuration.allow_origin='*'", "--Voila.ip=0.0.0.0"]
