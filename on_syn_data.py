import torch
import os
import yaml
import pandas as pd

from script import *

if __name__ == "__main__":    
    with open(f'configs.yaml', 'r') as file:
        config = transform_to_edict(yaml.safe_load(file)) 
    experiment = config.experiment
    model_ = config.model
    data_ = config.data

    width = experiment.print_width
    latex = experiment.latex  
     
    workspace, root_dir, log_dir = set_path(config,
        '--'.join([f"Model={'stochastic' if model_.autoencoder.stochastic else 'deterministic'}",
                   f"Com_fuser={model_.common_method}", 
                   f"Com_measure={config.loss.div_com}", 
                   f"Uni_mesure={config.loss.div_uni}"], 
                ), 
        use_format_time=False
    )
    data_.root_dir = workspace+"Data/Multi_view"
    data_.cuda = experiment.cuda
    data_.n_cross_validation=experiment.n_cross_validation
    
    manual_seed_all(experiment.seed) 

    dataset = 'synthetic'

    experiment.log_dir = os.path.join(log_dir, dataset)
    data_.update(dict(name=dataset, seed=experiment.seed))
    Data = LightData(data_, **config.dataloader)
    make_configs_consistent(Data, experiment, model_)
    model_.autoencoder.final_no_act=False
    model_.autoencoder.default_setting=False
    model_.autoencoder.n_latent_uni=1
    model_.autoencoder.n_latent_com=1
    model_.autoencoder.final_act='Tanh'
    Model= MultiView(**model_)
    myTrainer=Trainer(config, Model)
    print(f"\nfitting multi-view model for {dataset}")
    myTrainer.fit_unsupervised(Data)