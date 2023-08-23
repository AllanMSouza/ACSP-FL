import os
from optparse import OptionParser
import random


def add_server_info(clients, rounds, algorithm,  solution, dataset, model, poc, decay):
    server_str = f"  server:\n\
    image: 'allanmsouza/deev:server_fl'\n\
    container_name: fl_server\n\
    environment:\n\
      - SERVER_IP=0.0.0.0:9999\n\
      - NUM_CLIENTS={clients}\n\
      - NUM_ROUNDS={rounds}\n\
      - ALGORITHM={algorithm}\n\
      - POC={poc}\n\
      - SOLUTION_NAME={solution}\n\
      - DATASET={dataset}\n\
      - MODEL={model}\n\
      - DECAY={decay}\n\
    volumes:\n\
      - ./logs:/logs\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      placement:\n\
        constraints:\n\
          - node.role==manager\n\
    \n\n"

    return server_str


def add_client_info(client, model, client_selection, local_epochs, solution, algorithm,
					dataset, poc, decay, transmittion_threshold, personalization, shared_layers):
    client_str = f"  client-{client}:\n\
    image: 'allanmsouza/deev:client_fl'\n\
    environment:\n\
      - SERVER_IP=fl_server:9999\n\
      - CLIENT_ID={client}\n\
      - MODEL={model}\n\
      - CLIENT_SELECTION={client_selection}\n\
      - LOCAL_EPOCHS={local_epochs}\n\
      - SOLUTION_NAME={solution}\n\
      - ALGORITHM={algorithm}\n\
      - DATASET={dataset}\n\
      - POC={poc}\n\
      - DECAY={decay}\n\
      - TRANSMISSION_THRESHOLD={transmittion_threshold}\n\
      - PERSONALIZATION={personalization}\n\
      - SHARED_LAYERS={shared_layers}\n\
      - TIME2STARTMIN=0\n\
      - TIME2STARTMAX=30\n\
    volumes:\n\
      - ./logs:/logs\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      resources:\n\
        limits:\n\
          cpus: \"{random.uniform(0.01, 6)}\"\n\
      placement:\n\
        constraints:\n\
          - node.role==worker\n\
          \n\n"

    return client_str

def main():

    parser = OptionParser()

    parser.add_option("-c", "--clients",            dest="clients", default=0)
    parser.add_option("-m", "--model",              dest="model",   default='LR')
    parser.add_option("",   "--client-selection",   dest="client_selection",   default=False)
    parser.add_option("-e", "--local-epochs",       dest="local_epochs",   default=1)
    parser.add_option("-s", "--solution",           dest="solution",   default=None)
    parser.add_option("-a", "--algorithm",          dest="algorithm",  default=None)
    parser.add_option("-d", "--dataset",            dest="dataset",   default='UCIHAR')
    parser.add_option("-r", "--rounds",             dest="rounds",   default=100)
    parser.add_option("",   "--poc",                dest="poc",     default=0)
    parser.add_option("",   "--decay",              dest="decay",   default=0)
    parser.add_option("-t", "--threshold",          dest="transmittion_threshold",   default=0.2)
    parser.add_option("-p", "--personalization",    dest="personalization",   default=False)
    parser.add_option("",   "--shared-layers",      dest="shared_layers",   default=0)


    (opt, args) = parser.parse_args()

    with open(f'{opt.algorithm}-{opt.poc}-{opt.dataset}.yaml', 'a') as dockercompose_file:
        header = f"version: '3'\nservices:\n\n"

        dockercompose_file.write(header)

        server_str = add_server_info(opt.clients, opt.rounds, opt.algorithm, opt.solution, 
	    				opt.dataset, opt.model, opt.poc, opt.decay)

        dockercompose_file.write(server_str)

        for client in range(int(opt.clients)):
            client_str = add_client_info(client, opt.model, opt.client_selection, opt.local_epochs,
										 opt.solution, opt.algorithm, opt.dataset, opt.poc, opt.decay,
										 opt.transmittion_threshold, opt.personalization, opt.shared_layers)    

            dockercompose_file.write(client_str)




if __name__ == '__main__':
	main()