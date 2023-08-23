FROM python:3.9-slim

ADD data ./data
ADD client.py .


ADD dataset_utils.py .
ADD model_definition.py .

RUN pip install --upgrade pip
RUN pip install numpy "pandas<2.0.0" tensorflow flwr scikit-learn

CMD ["python", "-u", "client.py"]