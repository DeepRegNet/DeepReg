FROM tensorflow/tensorflow:latest-py3

# install
ADD /deepreg /app/deepreg
ADD requirements.txt /app
ADD setup.py /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -e .
