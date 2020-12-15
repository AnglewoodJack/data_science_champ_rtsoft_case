FROM python:3.7

WORKDIR /app

RUN pip install streamlit
RUN pip install numpy
RUN pip install pandas
RUN pip install plotly
RUN pip install matplotlib
RUN pip install holidays

EXPOSE 8051

COPY . /app

ENTRYPOINT ["streamlit","run"]

CMD ["app.py"]