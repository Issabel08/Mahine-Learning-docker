FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install  -r requirements.txt

VOLUME /usr/src/app/data  /usr/src/app/results/

COPY . .

CMD [ "python", "./ML.py" ]
