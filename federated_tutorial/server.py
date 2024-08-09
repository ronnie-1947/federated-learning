import hydra
from hydra.core.hydra_config import HydraConfig
from lib.federated import start_server
from pathlib import Path
import pickle

# Define the main function with the Hydra decorator
@hydra.main(config_path=None, config_name=None)

def main(cfg):
  history = start_server(num_rounds=4, host_addr='localhost:5050')
  
  #6. Save results
  save_path = HydraConfig.get().runtime.output_dir

  results_path = Path(save_path) / 'results.pkl'

  results = {'history': history}

  with open(str(results_path), 'wb') as h:
    pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
  
if __name__ == "__main__":
  main()