FROM ubuntu

RUN apt-get update

RUN apt-get install -y git python3 python3-pip

ADD API.py /API.py

ADD requirements.txt /requirements.txt

RUN pip install -r requirements.txt

RUN ["mkdir", "/html"]
RUN ["mkdir", "/model"]

COPY model/catboost_classifier.pkl /model/catboost_classifier.pkl

COPY ./html /html

CMD ["uvicorn", "--host", "0.0.0.0", "API:app"]