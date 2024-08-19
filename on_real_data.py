import torch
import os
import yaml
import pandas as pd

from script import LightData, MultiView, Trainer 
from script.utils import *
from script.data import valid_datasets

if __name__ == "__main__":    
    with open(f'configs.yaml', 'r') as file:
        config = transform_to_edict(yaml.safe_load(file)) 
    experiment = config.experiment
    model = config.model
    data = config.data

    # experiment.mode='debug'
    if experiment.mode=='debug':
        experiment.min_iters=1
        experiment.max_epochs=1
        experiment.n_cross_validation=2
        experiment.save_results=False
        experiment.view_latent=False

    data.cuda = experiment.cuda
    data.n_cross_validation=experiment.n_cross_validation

    width = experiment.print_width
    latex = experiment.latex  
    metrics= config.metrics
     
    workspace, root_dir, log_dir = set_path(config, "MonteCarloRuns", use_format_time=True)
    # workspace, root_dir, log_dir = set_path(config,"ViewLatent", use_format_time=False)
    data.root_dir = workspace+"Data/Multi_view"
    try:
        tuned_hyper = pd.read_csv(os.path.join(log_dir, "hyper_tune_results", f"tuned_hyper_params_all.csv"))
    except:
        pass

    statistics = {}
    seeds = torch.randperm(10000)[:experiment.n_cross_validation]
    for dataset in valid_datasets[:2]:
    # for dataset in [valid_datasets[-2]]:
        records={}
        data.update(dict(name=dataset, seed=experiment.seed))
        # Data = get_dataset(data)
        Data = LightData(data, **config.dataloader)
        make_configs_consistent(Data, experiment, model)        
        #load the optim weights for current dataset and update loss.weights
        if experiment.use_tuned_params:
            try:
                optim_weights = tuned_hyper.loc[tuned_hyper['dataset']==dataset].iloc[0].to_dict()     
                del optim_weights['accuracy'], optim_weights['dataset'], optim_weights['time']
                for key in ['ent_com', 'div_uni']:
                    if key in optim_weights.keys():
                        optim_weights[key] *= -1.0                
                config.loss.optim_weights.update(optim_weights)
                print("load hyper_parameters from Optuna tuned records.")
            except:
                pass
        print(f"\nfitting multi-view model for {dataset}")
        for i, seed in enumerate(seeds):            
            experiment.seed = seed.item()
            manual_seed_all(seed)         
            Data = Data.split_with_seed(seed=seed.item())
            experiment.log_dir = os.path.join(log_dir, dataset)
            Model= MultiView(**model)
            myTrainer=Trainer(config, Model)      
            myTrainer.set_fit_version(i+1)
            if not os.path.exists(os.path.join(myTrainer.log_dir, 'is_done')):
                # best_metrics, run_time = run(config, Model, Data)
                best_metrics, run_time = myTrainer.fit(Data)
            else:
                saved_results=loadmat(os.path.join(myTrainer.log_dir, 'results.mat'))
                index_best = torch.tensor(saved_results[experiment.basic_metric]).max(1)[1]
                best_metrics = {key: 100*saved_results[key][0][index_best] for key in config.metrics}    
                run_time = saved_results['run_time'].item() if saved_results.get('run_time') is not None else 0
                
            if i==0:
                records['run_num']=[f"veision_{i+1}"]
                records.update({key: [best_metrics[key]] for key in metrics})
                records['run_time']=[run_time]
            else:
                records['run_num'].append(f"veision_{i+1}")
                records.update({key: records[key]+[best_metrics[key]] for key in metrics})
                records['run_time'].append(run_time)

        print(f"\nresults of {dataset}:")
        statis=print_metrics(records, width, latex, os.path.join(experiment.log_dir, "records"))
        write_results_to_csv(statis, metrics, dataset, 'datasets', log_dir)
        statistics[dataset] = [stat.split(' ')[0].strip() for stat in statis.strip().split(' & ' if latex else ' '*3)]
    
    print(f"\nresults of all the datasets investigated: ")
    file = os.path.join(log_dir, f"statistics.{'tex' if latex else 'txt'}")
    keys=['datasets']+list(best_metrics.keys())+['run_time']
    print_statistics(statistics, keys, width, latex, file)


    # indics = list(statistics.keys())
    # metrics= list(best_metrics.keys())
    # df = pd.DataFrame({metric: [statistics[index][i] for index in indics] for i,metric in enumerate(metrics)}, 
    #                     index=[study for study in statistics.keys()],
    #                     )
    # df.index.name = 'datasets'
    # file=os.path.join(log_dir, f"statistics.csv")
    # df.to_csv(file, header=False if os.path.exists(file) else metrics, mode='a')
    