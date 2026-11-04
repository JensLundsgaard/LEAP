FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
