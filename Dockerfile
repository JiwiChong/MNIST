FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /workspace/MediWhale
ADD ./ /workspace/MediWhale
WORKDIR /workspace/MediWhale

RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py"]

