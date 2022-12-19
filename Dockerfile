FROM ubuntu:20.04
WORKDIR /app/
COPY . /app/
ENV AIRFLOW_HOME = "/app/airflow"
ENV DEBIAN_FRONTEND noninteractive
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT = 1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING = True
ENV TZ = Asia/Kolkata
RUN apt-get update -y && apt-get install python3-pip -y && apt-get install tzdata -y && apt install awscli -y 
RUN pip3 install -r requirements.txt
RUN airflow db init
RUN airflow users create -e suryanshgrover1999@gmail.com -f Suryansh -l Grover -p auntmay -r Admin -u admin
RUN chmod 777 start.sh
ENTRYPOINT ["/bin/sh"]
CMD ["start.sh"]