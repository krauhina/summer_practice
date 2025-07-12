FROM python:3.9-slim 
 
WORKDIR /app 
 
# Установка зависимостей 
RUN apt-get update && apt-get install -y \ 
    libgl1 \ 
    libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/* 
 
# Фиксируем версию numpy для совместимости 
RUN pip install numpy==1.23.5 
 
COPY ./app/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 
 
COPY app/ . 
 
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"] 
