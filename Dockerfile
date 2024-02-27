FROM python:3.9-slim

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libsndfile1 && \  
    rm -rf /var/lib/apt/lists/*


COPY . /usr/src/app

RUN apt-get update && apt-get install -y libx11-dev


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

ENV NAME World

CMD ["uvicorn", "comcopilotapp:app", "--host", "0.0.0.0", "--port", "8001"]
