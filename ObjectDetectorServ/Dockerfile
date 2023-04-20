# docker build -t object_detector_service_cpu .
# docker system prune
FROM python:3.10.5-slim-bullseye

RUN mkdir wd
WORKDIR /wd

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN mkdir src
COPY src/ ./src/

RUN python3 src/cache_model.py

# wsl --shutdown
# docker run -e PORT=80 -e MODEL_DEVICE=cpu -p 80:80 object_detector_service_cpu
# docker run object_detector_service_cpu sleep infinity
CMD exec gunicorn --workers=1 --threads=1 -b :$PORT src.main:app
#CMD python3 src/main.py