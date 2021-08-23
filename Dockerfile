# set base image (host OS)
FROM python:3.6

# set the working directory in the container
WORKDIR /meero

# copy the dependencies file to the working directory
COPY requirements.txt /meero

# install opencv
RUN apt-get update --fix-missing && apt-get install -y python3-opencv
RUN pip install opencv-python

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . /meero
