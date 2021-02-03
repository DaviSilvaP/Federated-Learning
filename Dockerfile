FROM tensorflow/tensorflow:latest-gpu

RUN pip install sklearn
RUN pip install fastapi
RUN pip install uvicorn

WORKDIR /tf/Scripts/
