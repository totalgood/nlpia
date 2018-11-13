FROM jupyter/datascience-notebook

USER root
WORKDIR /home/jovyan/nlpia
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /home/jovyan
RUN chown -R jovyan nlpia

EXPOSE 8888
