import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice
from math import sqrt, floor
from copy import deepcopy as copy
from os import makedirs, path

from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention as Attention

from cytoolz.itertoolz import sliding_window
slide = lambda series: sliding_window(2,series)


class FeatureFusioner(nn.Module):
    def __init__(self, feature_dim, attension_dim:int=0, fuse_method: str="cat"):
        super().__init__()
        self.fuse_method = fuse_method        
        if not self.fuse_method=='cat':
            self.attension_dim = attension_dim
            self.feature_dim = feature_dim        
            self.va=nn.Sequential(nn.Linear(feature_dim, feature_dim),nn.Tanh())
            self.ua=nn.Linear(feature_dim, feature_dim, bias=False)
            self.Lambda=nn.Softmax(dim=self.attension_dim)
        
    def forward(self, in_seqs):
        if self.fuse_method=='cat':
            return torch.cat(in_seqs,-1)
        else:
            if isinstance(in_seqs, list):
                in_seqs = torch.stack(in_seqs)
            out = (self.Lambda(self.va(in_seqs)*self.ua(in_seqs))*in_seqs).sum(self.attension_dim)
            return out

class FeatureMLP(nn.Module):
    def __init__(self, 
                 nodes=[256,256,256],
                 activation='ReLU',
                 use_batchnorm=True,
                 use_dropout=False,
                 prob=0.2,
                 final_activation='Sigmoid',
                 **kwargs):
        super(FeatureMLP, self).__init__()

        self.n_output = nodes[-1]
        models=[]
        for n_layer, (input,output) in enumerate(slide(nodes)):
            layers=[nn.Linear(input,output)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(output))
            if n_layer==len(nodes)-2: 
                if not kwargs.get('final_no_act'):
                    layers.append(getattr(nn, final_activation)())
            else:
                layers.append(getattr(nn, activation)())
            if use_dropout:
                layers.append(nn.Dropout1d(prob))
            models.append(nn.Sequential(*layers))
        self.model = nn.Sequential(*models)
    
    def forward(self, x):
        return self.model(x)
    
class Predictor(nn.Module):
    def __init__(self, 
                 feature_dim: int=512,
                 n_class: int = 10,
                 cls_hiddens: list=[],
                 use_batchnorm=True,
                 use_dropout=False,
                 prob=0.2,
                 **kwargs):
        super(Predictor, self).__init__()
        
        models=[]
        nodes = [feature_dim]+cls_hiddens+[n_class]
        for input,output in slide(nodes):
            layer=[]
            if use_dropout:
                layer.append(nn.Dropout(prob))
            layer.append(nn.Linear(input,output)) 
            if use_batchnorm:
                layer.append(nn.BatchNorm1d(output))   
            models.append(nn.Sequential(*layer))
        models.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*models)
        
    def forward(self, x):
        return self.model(x)

class ViewSpecific(nn.Module):
    com_uni_times=int(2)
    def __init__(self, 
                 name: str="view_0",
                 feature_dim: int=100,
                 encode_only: bool=False,
                 autoencoder: dict={},
                 **kwargs):
        super(ViewSpecific, self).__init__()
        self.name = name    
        self.encode_only = encode_only
        self.autoencoder_com = copy(autoencoder)
        self.autoencoder_uni = copy(autoencoder)
        self.dencoder_ = copy(autoencoder)
        if autoencoder.default_setting:
            self.unique_times = autoencoder.unique_times if autoencoder.get('unique_times') is not None else 5
            n_latent_uni = self.unique_times*autoencoder.n_class
            n_latent_com = self.com_uni_times*self.unique_times*autoencoder.n_class
        else:
            n_latent_uni = autoencoder.n_latent_uni
            n_latent_com = autoencoder.n_latent_com

        com_nodes = self.get_hidden_dims(feature_dim, n_latent_com)
        self.autoencoder_com.update(dict(nodes=com_nodes))
        self.common_encoder = FeatureMLP(**self.autoencoder_com)

        uni_nodes = self.get_hidden_dims(feature_dim, n_latent_uni)
        self.autoencoder_uni.update(dict(nodes=uni_nodes))
        self.unique_encoder = FeatureMLP(**self.autoencoder_uni)

        if not self.encode_only:
            decoder_nodes = [n+m for n,m in zip(com_nodes[::-1],uni_nodes[::-1])]
            decoder_nodes[-1] = feature_dim
            self.dencoder_.update(dict(nodes=decoder_nodes)
                                    )
            self.decoder = FeatureMLP(**self.dencoder_)

    def get_hidden_dims(self, input_dim, output_dim):
        return [input_dim,int(1.2*input_dim),int(0.5*input_dim), output_dim]

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        return self.common_encoder(batch), self.unique_encoder(batch)
    
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:    
        return self.decoder(embeddings)
            
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        common,unique=self.encode(batch)
        latent=torch.cat([common,unique],-1)
        outputs=dict(latent=latent,common=common,unique=unique)
        return  outputs
    
class MultiView(nn.Module):
    def __init__(self, 
                 n_view: int=2, 
                 n_class: int=10,
                 feature_dims: list=[],
                 encode_only: bool=False,
                 fuse_method: str="cat",
                 reconst_type: str='common',
                 common_method: str="random",
                 autoencoder: dict={},
                 predictor: dict={},
                 measure_structure: bool=False,
                 **kwargs) -> None:
        super(MultiView, self).__init__()
        self.n_view = n_view
        self.n_class = n_class
        self.encode_only = encode_only
        self.reconst_type = reconst_type
        self.common_method = common_method
        self.measure_structure = measure_structure

        self.autoencoders = nn.ModuleList([ViewSpecific(
                name=f"view_{n}",
                feature_dim=feature_dims[n],
                encode_only=encode_only,
                autoencoder=copy(autoencoder),
            ) for n in range(n_view)]
        )

        if fuse_method == "cat":
            self.n_latent = (self.autoencoders[0].common_encoder.n_output +
                n_view*self.autoencoders[0].unique_encoder.n_output)
        else:
            self.n_latent = self.autoencoders[0].common_encoder.n_output+self.autoencoders[0].unique_encoder.n_output
        predictor['feature_dim'] = int(self.n_latent)        
        self.classifier=Predictor(**predictor)
        self.fusioner=FeatureFusioner(self.n_latent,fuse_method=fuse_method)
        if common_method=='align':
            self.fusioner_common=FeatureFusioner(self.autoencoders[0].common_encoder.n_output,fuse_method='attension')   
    
    def save_state_dict(self, filename):
        makedirs(path.dirname(filename), exist_ok=True)
        self.init_check_point = filename
        if not path.exists(filename):
            torch.save(self.state_dict(), filename)

    def reset_weights(self):
        try:
            self.load_state_dict(torch.load(self.init_check_point))
        except:
            pass

    def feature_extract(self, views):        
        return [encoder(view) for view,encoder in zip(views, self.autoencoders)]
        #debug
        # outputs=[]
        # for view,encoder in zip(views, self.autoencoders):
        #     outputs.append(encoder(view))
        # return outputs
    
    def image_reconstruct(self, latents, uniques, common):
        if self.reconst_type=='common':
            hiddens = [torch.cat([unique, common],-1) for unique in uniques]
        else:
            hiddens = latents
        return [autoencoder.decode(latent) for autoencoder,latent in zip(self.autoencoders, hiddens)]
    
    def classify(self, latents):
        return self.classifier(latents)
    
    def forward(self, views, labels):
        outputs = self.feature_extract(views)

        latents = [output['latent'] for output in outputs]
        commons = [output['common'] for output in outputs]
        uniques = [output['unique'] for output in outputs]

        if self.common_method=='random':# random choice common            
            common = commons[choice([i for i in range(self.n_view)])]
        else:# attension fusion
            common = self.fusioner_common(commons)

        latents_fused=torch.cat([self.fusioner(uniques), common],-1)
        predict=self.classify(latents_fused)

        multi_out = dict(
            common=common,
            latent=latents_fused,
            predict=predict,
            labels=labels,
            commons=commons,
            uniques=uniques,
            latents=latents,
            inputs=views,
        )
        if not self.encode_only:
            multi_out.update(dict(recons = self.image_reconstruct(latents, uniques, common)))
        
        return multi_out
    
    def loss_function(self, multi_out:dict, sample_weights:torch.Tensor, **kwargs):
        """
        multi_out: keys: 
            common,
            latent,
            predict,
            commons,
            uniques,
            latents,
            labels,
            reconsts <-- optional
            inputs
        """
        weights = kwargs['weights']
        inputs = multi_out['inputs']
        reduction='mean' if sample_weights is None else 'none'
        loss_dict={}

        if 'ce' in weights.keys():
            ce = kwargs['ce'](multi_out['predict'], multi_out['labels'], reduction=reduction)
            loss_dict.update({'ce': ce if sample_weights is None else (sample_weights*ce).sum()})

        if (not self.encode_only) and ('rec' in weights.keys()):
            recons = multi_out['recons']
            reconsts = [F.mse_loss(input, recon, reduction=reduction) for input, recon in zip(inputs, recons)]
            if reduction == 'none':
                reconsts = [(sample_weights.view(1,-1).matmul(rec).mean(-1)).sum() for rec in reconsts]
            loss_dict.update({f'rec_{i}': rec.item() for i,rec in enumerate(reconsts)}) 
            loss_dict.update(dict(rec = sum(reconsts)/self.n_view))
        
        
        sigmas = kwargs.get('sigmas')
        if 'div_com' in weights.keys():
            div_measure = kwargs['div_com']
            sources=multi_out['commons']
            if self.common_method=='align':
                sources=sources+[multi_out['common']]
            if div_measure.__name__=="bary_center_dissimilarity":            
                bcd, divs = div_measure(*sources, 
                                        bary_center=multi_out['common'] if self.common_method=='align' else None,
                                        measure = kwargs.get('bcd_measure'))
                loss_dict['div_com'] = bcd
                loss_dict.update({f'div_com_{i}': div.item() for i,div in enumerate(divs)})
            else:
                loss_dict['div_com'], loss_dict['sigma_div_com'] = div_measure(*sources, sigmas=sigmas)
        
        if 'div_uni' in weights.keys():
            div_measure = kwargs['div_uni']
            if self.measure_structure:
                to_measure = [uni.transpose(0,1) for uni in multi_out['uniques']]
            else:
                to_measure = multi_out['uniques']
                
            if self.autoencoders[0].com_uni_times==1:
                to_measure.append(multi_out['common'])

            loss_dict['div_uni'], loss_dict['sigma_div_uni'] = div_measure(*to_measure, sigmas=sigmas)
            
        if 'ent_com' in weights.keys():
            entropy_measure=kwargs['ent_com']
            loss_dict['ent_com'], loss_dict['sigma_ent_com'] = entropy_measure(multi_out['common'])

        if 'ent_all' in weights.keys():
            entropy_measure=kwargs['ent_all']
            loss_dict['ent_all'], loss_dict['sigma_ent_all'] = entropy_measure(multi_out['latent'])
            
        loss_dict['loss'] = sum([weight*loss_dict[key] for key,weight in weights.items()])
        loss_dict.update({key: loss_dict[key].item() for key in list(weights.keys())})
        if not kwargs['is_training']:
            loss_dict['loss'] = loss_dict['loss'].item()

        return loss_dict

    def metric_function(self, preds, labels, metric: callable=None):
        return metric(preds.argmax(1), labels)