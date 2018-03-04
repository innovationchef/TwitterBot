FROM python:2.7

LABEL maintainer = Ankit Lohani: lohani.1575@gmail.com

RUN mkdir /src
WORKDIR /src

RUN apt-get update
COPY requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt --proxy=172.16.2.30:8080

COPY . /src
CMD ["python", "/test.py" ]
