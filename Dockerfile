FROM ubuntu:18.04

WORKDIR /usr/src/kmeans_server

COPY . .
RUN apt clean
RUN apt-get update
RUN apt-get install python3.7 python3-pip zlib1g-dev libjpeg-dev -y
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /usr/src/kmeans_server/webapp

CMD ["python3","webapp.py"]
