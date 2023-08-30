# ACSP-FL: Adaptive Client Selection with Personalization for Communication Efficient Federated Learning

**Abstract**: 

## Available Datasets

The following datasets are available:
- CIFAR-10
- MNIST
- Motion sense
- UCI-HAR
- ExtraSensory

## Parâmetros para gerar docker-compose-file:
- `--clients` `-c`: Quantidade total de clientes
- `--model` `-m`: Modelo de ML/DL para ser utilizado no treinamento (e.g., DNN, CNN, or Logistic Regression)
- `--client-selection` `-`: Método para seleção de clientes (e.g., POC, DEEV)
- `--dataset` `-d`: Dataset para ser utilizado no treinamento (e.g., MNIST, CIFAR10)
- `--local-epochs` `-e`:  Quantidade de épocas locais de treinamento
- `--rounds` `-r`: Número de rodadas de comunicação para o treinamento
- `--poc` `-`: Porcentagem de clientes para ser selecionados no Power-of-Choice
- `--decay` `-`: Parâmetros para decaimento no DEEV

É importante gerar novas imagens tanto para o Cliente quanto para o Servidor com o Dockerfile de ambos os diretórios. Em seguida, substitua a imagem no script `create_dockercompose.py`

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
@inproceedings{acsp-fl,
 
}

```
