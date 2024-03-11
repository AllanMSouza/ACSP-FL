# ACSP-FL: Adaptive Client Selection with Personalization for Communication Efficient Federated Learning

**Abstract**: *Federated Learning (FL) is a distributed approach to collaboratively training machine learning models. FL requires a high level of communication between the devices and a central server, thus imposing several challenges, including communication bottlenecks and network scalability. This article introduces ACSP-FL, a solution to reduce the overall communication and computation costs for training a model in FL environments. ACSP-FL employs a client selection strategy that dynamically adapts the number of devices training the model and the number of rounds required to achieve convergence. Moreover, ACSP-FL enables model personalization to improve clients performance. A use case based on human activity recognition datasets aims to show the impact and benefits of ACSP-FL when compared to state-of-the-art approaches. Experimental evaluations show that ACSP-FL minimizes the overall communication and computation overheads to train a model and converges the system efficiently. In particular, ACSP-FL reduces communication up to 95% compared to literature approaches while providing good convergence even in scenarios where data is distributed differently, non-independent and identical way between client devices.*

## Available Datasets

The following datasets are available:
- CIFAR-10
- MNIST
- Motion sense
- UCI-HAR
- ExtraSensory

## docker-compose-file generator parameters:
- `--clients` `-c`: number of clients
- `--model` `-m`: model to be used in the simulation (e.g., DNN, CNN, or Logistic Regression)
- `--client-selection` `-`: client selection method 
- `--dataset` `-d`: Dataset to be used by the clients
- `--local-epochs` `-e`:  number of local epochs in each client
- `--rounds` `-r`: total number of communication rounds
- `--poc` `-`: percentagem of clients to be selected using PoC 
- `--decay` `-`: decay factor for DEEV and ACSP-FL
- `--personalization` `-`: enables personalization and partial model sharing 
- `--shared-layers` `-`: number of layers to be shared in the personalization phase

**tip**: The ACSP-FL solution is the DEEV + Personalization enabled

It is important to generate new images for both the Client and the Server with the Dockerfile from both directories. Then replace the image in the `create_dockercompose.py` script

## Creating Docker compose files:
```python
python create_dockercompose.py --client-selection='DEEV' --dataset='MNIST' 
--model='DNN' --epochs=1 --round=10 --clients=50 
```

## How to execute?
```shell
docker compose -f <compose-file.yaml> --compatibility up 
```
## How to cite?
```bibtex
@article{acsp-fl,
title = {Adaptive client selection with personalization for communication efficient Federated Learning},
journal = {Ad Hoc Networks},
volume = {157},
pages = {103462},
year = {2024},
issn = {1570-8705},
doi = {https://doi.org/10.1016/j.adhoc.2024.103462},
url = {https://www.sciencedirect.com/science/article/pii/S1570870524000738},
author = {Allan M. {de Souza} and Filipe Maciel and Joahannes B.D. {da Costa} and Luiz F. Bittencourt and Eduardo Cerqueira and Antonio A.F. Loureiro and Leandro A. Villas},
}
```
