import torch, os
import tqdm
from typing import List, Optional
from tensorboardX import SummaryWriter
from time import time

from .loss import Loss

from .utils import (get_optimizer, 
                    get_scheduler, 
                    update_callback, 
                    savemat,
                    loadmat,
                    visualize_results,
                    visualize_synthetic,
                    classification_results,
                    save_hyperparameters,
                    cat_listed_dict,
                    copy,
                    edict,
                    EarlyStop,
                    myEventLoader)
class Trainer():
    def __init__(self, configs, model) -> None:
        self.configs= configs
        # set all attributes of experiment to be private attributes
        self.__dict__.update(copy(configs.experiment))   
        self.model = model  
        if self.cuda:
            self.model=self.model.cuda()
        
        loss_caller=copy(configs.loss)
        loss_caller.update(edict({key: Loss[val] for key, val in loss_caller.items() 
                                 if isinstance(val, str) and not key=='rec'}))
        self.loss = loss_caller
        self.losses = sorted(list(configs.loss.weights.keys())+['loss'])

        self.metrics = configs.metrics

        self.optimizer = get_optimizer(configs.optimizer)(model)
        self.scheduler = get_scheduler(configs.scheduler)(self.optimizer)

        self.root_dir = copy(self.log_dir)

    def set_fit_version(self, version:int=1):
        self.version = version
        self.log_dir = os.path.join(self.root_dir, f"version_{version}")
        if self.save_results:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(logdir=self.log_dir)
            self.eventloader = myEventLoader(self.writer.logdir)
            save_hyperparameters(self.configs, os.path.join(self.log_dir, "check_points", "configs.yaml"))
            self.model.save_state_dict(os.path.join(self.log_dir, "check_points", "init_weights.pt"))
    
    def fit(self, DataModule):  
        train_dataloader = DataModule.train_dataloader()
        val_dataloader   = DataModule.val_dataloader()
        results = []
        postfix = {f'{self.basic_metric}_best': "%.2f" % 0.0,
                    'accuracy': "%.2f" % 0.0,
                    'f1-score': "%.2f" % 0.0,
                    'recall': "%.2f" % 0.0,
                    'precision': "%.2f" % 0.0}
        iter = tqdm.tqdm(range(self.max_epochs),
        # iter = tqdm.gui.tqdm(range(self.max_epochs),
                    disable=self.silent,
                    postfix=postfix,
                    desc=f'run version {self.version}',
                    unit='epoch',
                    )
         
        best_metric, best_metrics, epoch_results = 0.0, {}, {}
        if self.plot_mode=='step':
            resluts_accumulate = {}
        start = time()
        earlystop_triggered = False
        if self.enable_earlystop:
            self.earlystop = EarlyStop(patience=30, direction="maximize", limit=1.0)  
        for epoch in iter:
            self.epoch = epoch
            self.train_epoch(train_dataloader)
            self.val_epoch(val_dataloader)
            epoch_results.update(self.epoch_results_train)
            epoch_results.update(self.epoch_results_val)
            epoch_results.update(self.report)
            
            postfix.update({key: f"{val*100:.2f}" for key,val in self.report.items()})
            if best_metric<self.report[self.basic_metric]:
                best_metric=self.report[self.basic_metric]
                postfix.update({f'{self.basic_metric}_best': f"{best_metric*100:.2f}"})
                best_metrics=copy(postfix)     

            if self.save_results:    
                results.append(epoch_results)
            
                if self.plot_mode=='step':
                    if epoch==0:
                        resluts_accumulate = self.log_step
                        resluts_accumulate.update(
                            {key: [val] for key,val in self.report.items()})
                    else:
                        for key,val in self.log_step.items():
                            resluts_accumulate[key].extend(val)
                        for key,val in self.report.items():
                            resluts_accumulate[key].append(val)
                update_callback(self.writer, epoch_results['steps'], 
                    {f"{key}_epoch": val for key,val in epoch_results.items() 
                     if key not in self.metrics})       

            iter.set_postfix(postfix if epoch<self.max_epochs else best_metrics)   
            if self.enable_earlystop and self.earlystop(self.report[self.basic_metric]):    
                iter.set_postfix(best_metrics)        
                earlystop_triggered = True
                break
        
        if earlystop_triggered:        
            print('experiment run {}/{} {} for {} @ epoch {}.\n'.format(
                self.version,self.n_cross_validation,self.earlystop.info,self.basic_metric,epoch+1))
        best_metrics = {key: float(val) for key,val in best_metrics.items() if not 'best' in key}
        # best_metrics = {key: max(save_dict[key]) for key in configs.metrics}
        run_time = time()-start
        
        if self.save_results:
            results = {key: [res[key] for res in results] for key in epoch_results.keys()}
            if self.plot_mode=='step':
                results=resluts_accumulate
            try:
                self.eventloader.events_to_mat(file_num=0)
            except:
                pass
            self.writer.close()        
            # show_name = "_".join(self.log_dir.split(os.sep)[-2:])    
            show_name = self.log_dir.split(os.sep)[-1]    
            visualize_results(results, 
                            show_keys=self.losses, show_name=show_name+'_loss', path=self.log_dir, single=False)
            visualize_results(results, show_keys=self.metrics, show_name=show_name+'_metric', path=self.log_dir)
            
            savemat(f"{self.writer.logdir}/results.mat", results)

            with open(os.path.join(self.log_dir, 'is_done'), 'w') as f:
                f.write('done')

            del results, resluts_accumulate
        
        return (best_metrics, run_time)
        
    def train_epoch(self, dataloader):
        iter_per_epoch = len(dataloader)
        epoch = self.epoch

        if epoch==0:
            self.postfix_train={key: "%.6f"%0 for key in self.losses}
        iter = tqdm.tqdm(dataloader,
                    disable=self.silent,
                    postfix=self.postfix_train,
                    desc='trainning loop',
                    unit="batch",
                    leave=False)
        self.model.train()
        start=time()
        for index, (features, labels, weights) in enumerate(iter):
            if self.cuda and not labels.is_cuda:
                features = [view.cuda(non_blocking=True) for view in features]
                labels = labels.cuda(non_blocking=True)
                weights = weights.cuda(non_blocking=True)
            output = self.model(features, labels)
            loss_dict = self.model.loss_function(output, 
                        weights if self.enable_sample_weight else None, 
                        is_training=True, **self.loss)
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step(closure=None)
            steps = epoch*iter_per_epoch+index           
            loss_dict['loss'] = loss_dict['loss'].item() 
            
            log_dict=dict(steps=steps, 
                          lr=self.optimizer.param_groups[0]["lr"],
                          **loss_dict)
            if self.save_results and steps%self.update_freq==0:
                update_callback(self.writer, steps, log_dict,)       
            if index==0:
                self.log_step = {key: [val] for key,val in log_dict.items()}
            else:
                self.log_step = {key: self.log_step[key]+[val] for key,val in log_dict.items()}
            
            # postfix=dict(step=f"{steps:4d}/{iter_per_epoch*runner.max_epochs:5d}")
            self.postfix_train.update({key: "%.6f" % loss_dict[key]  for key in self.losses})
            iter.set_postfix(self.postfix_train)
        try:
            self.scheduler.step()
        except:
            pass
        
        self.epoch_results_train=dict(t_train=time()-start, steps=steps,            
            **{key: sum(val)/iter_per_epoch for key,val in self.log_step.items() if key in self.losses})
        
        self.output = edict(**output)

        return self.epoch_results_train

    def val_epoch(self, dataloader):
        iter_per_epoch = len(dataloader)
        epoch = self.epoch
        
        if epoch==0:
            self.postfix_val={key: "%.6f"%0 for key in self.losses}
        iter = tqdm.tqdm(dataloader,
                    disable=self.silent,
                    postfix=self.postfix_val,
                    desc='validation loop',
                    unit="batch",
                    leave=False)
        self.model.eval()
        start=time()
        preds,labels=[],[]
        for index, (feature, label, weight) in enumerate(iter):
            if self.cuda and not label.is_cuda:
                feature = [view.cuda(non_blocking=True) for view in feature]
                label = label.cuda(non_blocking=True)
                weight = weight.cuda(non_blocking=True)
            output = self.model(feature, label)
            loss_dict = self.model.loss_function(output, 
                            weight if self.enable_sample_weight else None, 
                            is_training=False, **self.loss)
            preds.extend(list(output['predict'].argmax(1).detach().cpu().numpy()))
            labels.extend(list(label.detach().cpu().numpy()))
            steps = epoch*iter_per_epoch+index  
            if self.save_results:
                update_callback(self.writer, steps, {f"{key}_val": val for key,val in loss_dict.items()})
                
            if index==0:
                self.log_step_val = {key: [val] for key,val in loss_dict.items()}
            else:
                self.log_step_val = {key: self.log_step_val[key]+[val] for key,val in loss_dict.items()}
            
            self.postfix_val.update({key: "%.6f" % loss_dict[key]  for key in self.losses})
            iter.set_postfix(self.postfix_val)

        self.epoch_results_val = dict(t_val=time()-start,
                    **{f'{key}_val': sum(val)/iter_per_epoch for key,val in self.log_step_val.items()})
        self.report = classification_results(labels, preds, self.metrics)    
        return self.report

    def fit_unsupervised(self, DataModule):
        dataloader = DataModule.all_dataloader()
        epoch_results = []
        iter = tqdm.tqdm(range(self.max_epochs),
                    desc='synthetic trainning',
                    unit="epoch",)
        for epoch in iter:
            self.epoch = epoch
            self.train_epoch(dataloader)
            if self.save_results:
                epoch_results.append(self.epoch_results_train)
                update_callback(self.writer, self.epoch_results_train['steps'], 
                    {f"{key}_epoch": val for key,val in self.epoch_results_train.items()})       
        if self.save_results: 
            visualize_results(cat_listed_dict(epoch_results), 
                              show_keys=self.losses, 
                              show_name='loss', 
                              path=self.root_dir)
            del epoch_results
            
            if DataModule.name=="synthetic":
                data_to_visualize=edict(
                    T = DataModule.data.t.T,
                    common_orig=DataModule.data.com.T,
                    noise = DataModule.data.noise.T,
                    common_trained=self.output.common.detach().cpu().numpy(),
                    uniques_orig=[uni.T for uni in DataModule.data.unis],
                    uniques_trained=[uni.detach().cpu().numpy() for uni in self.output.uniques],
                    commons_trained=[com.detach().cpu().numpy() for com in self.output.commons]
                )

                visualize_synthetic(data_to_visualize, path=self.root_dir)
                