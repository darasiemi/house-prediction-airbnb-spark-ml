FROM python:3.9.1

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean  

ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION

ENV PATH=$PATH:$SPARK_HOME/bin

RUN mkdir -p /opt/spark && \
    cd /opt/spark && \
    wget https://dlcdn.apache.org/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz && \
    tar -xvf spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz && \
    rm spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz 
    
WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN mkdir -p /app/data

COPY spark_machine_learning.py ./
COPY data/sf-airbnb-clean-100p.parquet.zip /app/data/

RUN unzip /app/data/sf-airbnb-clean-100p.parquet.zip -d /app/data/ && rm /app/data/sf-airbnb-clean-100p.parquet.zip

ENTRYPOINT ["python", "spark_machine_learning.py"]
