ARG PORT = 443
FROM cypress/browsers:latest

RUN apt-get install python 3.10.8 -y
RUN echpo $(python3 -m site --user-base)
COPY requirements.txt .
ENV PATH /home/root/.local/bin:${PATH}
RUN apt-get update && apt-get install -y python3-pip && pip install -r requirements.txt
COPY . .
CMD uvocorn main: app -- host 0.0.0.0 --port $PORT
