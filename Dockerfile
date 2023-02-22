FROM tensorflow/tensorflow:2.6.0-gpu
RUN apt-get update
# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt .
# Install the requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get install -y python3.8 \
    && ln -s /usr/bin/python3.8

COPY --from=builder . /usr/src/app
COPY . /usr/src/app


ENV HOME=/usr/src/app

CMD ["python", "app.py"]
EXPOSE 8080