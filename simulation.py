import flwr as fl

from Client.client import FedClient
from Server.server import FedServer
from Client.dataset_utils import ManageDatasets


NCLIENTS = 30
DATASET  = 'UCIHAR'
#docker compose -f .\DEEV-0.5-ExtraSensory.yaml --compatibility up
def create_client(cid):
        return FedClient(cid=cid, n_clients=NCLIENTS, epochs=1, 
				 model_name             = 'DNN', 
				 client_selection       = False, 
				 solution_name          = 'TESTE', 
				 aggregation_method     = 'DEEV',
				 dataset                = DATASET,
				 perc_of_clients        = 0,
				 decay                  = 0.005,
				 transmittion_threshold = -1)

class SimulationAdhoc():

    def __init__(self) -> None:
        pass

    
    
    def create_server(self):
        return FedServer('DEEV', 1, NCLIENTS, 
					decay=0.005, 
                    perc_of_clients=0, 
                    dataset=DATASET, 
                    solution_name='DEEV', 
                    model_name='DNN')

    def run_simulation(self):
        ray_args = {
			"include_dashboard"   : False,
			"ignore_reinit_error" : True
		}
        fl.simulation.start_simulation(
            client_fn     = create_client,
            num_clients   = NCLIENTS,
            config        = fl.server.ServerConfig(num_rounds=100),
            strategy      = self.create_server(),
            ray_init_args = ray_args
        )


SimulationAdhoc().run_simulation()

