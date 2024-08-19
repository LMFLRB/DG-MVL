import torch
import torch.distributions as T_dists

import warnings
from typing import Union
from torch.nn.functional import cosine_similarity

from numpy import ndarray
from torch import Tensor
from copy import deepcopy as copy
from math import sqrt, pi
EPSILON = 1E-20

warnings.filterwarnings("ignore")
EPS = 1.0e-20


################### basic functions ####################
def atleast_epsilon(X, eps=1.0e-15):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return torch.where(X < eps, X.new_tensor(eps), X)

def clip(input, min=1.0e-10, max=float(torch.inf)):
    return torch.clip(input, min, max)    

def get_kernel_size(x, method="median"):
    """
    automatically estimate kernelsize.
    """
    x_ = x.flatten(1)
    x_ = x_.max(1)[0]-x_.min(1)[0]
    n = len(x_)
    sample_std = torch.std(x_, unbiased=True)

    if method == 'silverman':
        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        z=torch.sort(x_)[0]
        iqr = torch.quantile(z, 0.75) - torch.quantile(z, 0.25)
        bandwidth = 0.9 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)

    elif method.lower() == 'gauss-optimal':
        bandwidth = 1.06 * sample_std * (n ** -0.2)        

    elif method.lower() == 'median':
        bandwidth = 0.15 * torch.median(x_)

    elif method.lower() == 'mean':
        bandwidth = x_.mean()

    else:        
        raise ValueError(f"Invalid method selected: {method}.")

    return clip(bandwidth,1.e-2,1.e4).item()**2

def kernel_size_estimation(dist, min=1.0e-2, max=1.0e4):
    index_triu = torch.triu_indices(*dist.shape, 1)
    dist_triu = dist[index_triu[0],index_triu[1]]
    sigma = dist_triu.abs().mean()
    return clip(sigma,min,max).item()

def p_dist_2(x, y):
    # x, y should be with the same flatten(1) dimensional
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    x_norm2 = torch.sum(x**2, -1).reshape((-1, 1))
    y_norm2 = torch.sum(y**2, -1).reshape((1, -1))
    dist = x_norm2 + y_norm2 - 2*torch.mm(x, y.t())
    return torch.where(dist<0.0, torch.zeros(1).to(dist.device), dist)

def calculate_gram(*data, sigma=None, **kwargs):
    if len(data) == 1:
        x, y = data[0], data[0]
    elif len(data) == 2:
        x, y = data[0], data[1]
    else:
        print('size of input not match')
        return []
    dist = p_dist_2(x, y)    
    if sigma is None:
        # sigma = get_kernel_size(torch.cat([x,y]))*10
        sigma = kernel_size_estimation(dist)
    scale = 1./sqrt(2.*pi*sigma)
    return atleast_epsilon(scale*torch.exp(-.50*dist / sigma), 1e-10), sigma

def calculate_gram_transversal(*domains:Union[list,tuple], 
                      sigmas: Union[Tensor,ndarray,float]=None,
                      **kwargs)->Tensor:
    
    """ Calculate the gram-matrix with for loop transversallly, 
        i.e., get the pairwise-distance and exponetial matrix group by group
    """
    dn, bn = len(domains), len(domains[0])
    PD = torch.zeros([dn,dn,bn,bn], device=domains[0].device)
    out_sigmas=torch.zeros([dn,dn], device=domains[0].device)
    for t in range(dn):
        for k in range(t+1):
            pd_tk = p_dist_2(domains[t], domains[k])
            if sigmas is not None:
                sigma = sigmas
            else:
                # sigma = get_kernel_size(torch.cat([domains[t],domains[k]],0), "silverman")**2
                sigma = kernel_size_estimation(pd_tk)
            scale = 1./sqrt(2.*pi*sigma)
            PD[t,k,...] = PD[k,t,...] = (-pd_tk/sigma/2).exp()*scale
            out_sigmas[t,k] = out_sigmas[k,t] = sigma
    
    return PD, out_sigmas   

def renyi_entropy(x, sigma=None, alpha=1.001):
    k,sigma = calculate_gram(x, x, sigma=sigma)
    k = k/torch.trace(k)
    # eigv = torch.abs(torch.linalg.eigh(k)[0])
    try:
        eigv = torch.abs(torch.linalg.eigh(k)[0])
    except:
        eigv = torch.diag(torch.eye(k.shape[0]))
    entropy = (1/(1-alpha))*torch.log2((eigv**alpha).sum(-1))
    return entropy,sigma

def joint_entropy(x, y, s_x, s_y, alpha=1.001):
    x = calculate_gram(x, sigma=s_x)[0]
    y = calculate_gram(y, sigma=s_y)[0]
    k = torch.mul(x, y)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

def entropy_from_gram(k, alpha=1.001):
    k = k/torch.trace(k)
    # eigv = torch.abs(torch.linalg.eigh(k)[0])
    try:
        eigv = torch.abs(torch.linalg.eigh(k)[0])
    except:
        eigv = torch.diag(torch.eye(k.shape[0]))
    entropy = (1/(1-alpha))*torch.log2((eigv**alpha).sum(-1))
    return entropy

def mee(*data, sigma=None, **kwargs):
    if len(data) == 1:
        x, y = data[0], data[0]
    elif len(data) == 2:
        x, y = data[0], data[1]
    else:
        print('size of input not match')
        return []
    dist = p_dist_2(x, y)    
    if sigma is None:
        # sigma = get_kernel_size(dist_triu)**2
        sigma = kernel_size_estimation(dist)*2.0
    k=(-dist / sigma).exp()
    k = k/torch.trace(k)
    return -k.mean().log(), sigma
    
def mutual_information(x,y,sigma_x=None,sigma_y=None, alpha=1.001):
    if sigma_x is None:
        kxx, sigma_x = calculate_gram(x)
    else:
        kxx = calculate_gram(x, sigma_x)        
    if sigma_y is None:
        kyy, sigma_y = calculate_gram(y)
    else:
        kyy = calculate_gram(y, sigma_y)

    kxy = torch.mul(kxx, kyy)
    kxy = kxy/torch.trace(kxy)

    Hx = entropy_from_gram(kxx)
    Hy = entropy_from_gram(kyy)
    Hxy = entropy_from_gram(kxy)
    Ixy = Hx+Hy-Hxy
    Ixy = Ixy/(torch.max(Hx,Hy)+1e-16)

    return Ixy

def mhsic(samplesets:Union[list,tuple,Tensor], 
               sigmas: Union[Tensor,ndarray,float]=None, 
               **kwargs)->Tensor:
    from numpy import arange, triu_indices
    dn, bn = len(samplesets), len(samplesets[0])
    # K = calculate_gram_comprehensive(samplesets,sigmas,**kwargs)[0]
    K = calculate_gram_transversal(samplesets,sigmas=sigmas,**kwargs)[0]

    idx_diag = arange(dn)
    idx_triu = triu_indices(dn, 1)
    Diag = K[idx_diag, idx_diag, ...]
    Cross= K[idx_triu[0], idx_triu[1], ...]
    mhsic = Diag.prod(0).mean() + Diag.mean([-1,-2]).prod() - 2.*Cross.mean([-1,-2]).prod()
    return mhsic

def row_prod_except(A):    
    res = torch.cat([torch.cat([A[:,:j], A[:,j+1:]],1).prod(1,True) for j in range(A.shape[1])],1)
    return atleast_epsilon(res, EPS)

############## kld #####################################
def kld(X:Tensor, Y:Tensor, sigma=None, **kwargs):
    m=len(X)
    k = calculate_gram(torch.cat([X, Y],0), sigma=sigma)[0]     
    pxx = atleast_epsilon(k[:m,:m].mean(-1))
    pyy = atleast_epsilon(k[m:,m:].mean(-1))
    pxy = atleast_epsilon(k[:m,m:].mean(-1))
    pyx = atleast_epsilon(k[m:,:m].mean(-1))

    # return (pxx.log()-pxy.log()).mean()
    return (pxx.log()-pyx.log()).mean() + (pyy.log()-pxy.log()).mean()

############## GJSD #####################################
def gjsd(*domains:Union[list,tuple,Tensor], 
         sigmas: Union[Tensor,ndarray,float]=None,
         **kwargs) -> Tensor:
    dn = len(domains)
    if 'weights' in kwargs:
        B = Tensor(kwargs['weights']).to(domains[0].device)
        if len(B)==2 and dn>2:
            B = Tensor([B[0]/(dn-1)]*(dn-1)+[B[1]]).to(domains[0].device)/B.sum()        
    else:
        B = torch.ones([dn,]).to(domains[0].device)/dn
    # gram_mat = calculate_gram_comprehensive(domains, sigmas, **kwargs)[0]
    gram_mat = calculate_gram_transversal(*domains, sigmas=sigmas, **kwargs)[0]
       
    tri_d = torch.arange(dn)
    
    PDF = gram_mat.mean(-1)
    power_PDF = PDF[tri_d,tri_d,:]
    cross_entropy = -atleast_epsilon(B.view(1,-1).matmul(PDF)).log().mean()   
    power_entropy = -atleast_epsilon(power_PDF).log().mean(-1).matmul(B)

    return (cross_entropy - power_entropy)

############ csd #######################
def csd(X:Tensor, Y:Tensor, sigma=None, **kwargs)->Tensor:
    m=len(X)
    k, sigma = calculate_gram(torch.cat([X, Y],0), sigma=sigma)    
    p1_square = k[:m,:m].mean() #etismate p1 with sampleset1
    p2_square = k[m:,m:].mean() #etismate p2 with sampleset2
    p12_cross = k[:m,m:].mean() #etismate p1 with sampleset2
    cs = (p1_square.log2()+p2_square.log2())*0.5 - p12_cross.log2()
    return cs

def csd_QMI(x,y,sigma = None):
    """
    x: NxD
    y: NxD
    Kx: NxN
    ky: NxN
    """
    
    N = x.shape[0]
    #print(N)
    if not sigma:
        sigma_x = 10*kernel_size_estimation(p_dist_2(x,x).abs())
        sigma_y = 10*kernel_size_estimation(p_dist_2(y,y).abs())
       
        Kx = calculate_gram(x,x,sigma=sigma_x)[0]
        Ky = calculate_gram(y,y,sigma=sigma_y)[0]
    
    else:
        Kx = calculate_gram(x,x,sigma=sigma_x)[0]
        Ky = calculate_gram(y,y,sigma=sigma_y)[0]
    
    #first term
    self_term1 = torch.trace(Kx@Ky.T)/(N**2)
    
    #second term  
    self_term2 = (torch.sum(Kx)*torch.sum(Ky))/(N**4)
    
    #third term
    term_a = torch.ones(1,N).to(x.device)
    term_b = torch.ones(N,1).to(x.device)
    cross_term = (term_a@Kx.T@Ky@term_b)/(N**3)
    CS_QMI = torch.log2(self_term1) + torch.log2(self_term2)-2*torch.log2(cross_term)
    
    return CS_QMI

def ccsd(Xp:Tensor, Xq:Tensor, Yp:Tensor, Yq:Tensor, 
         sigmas=None, **kwargs)->Tensor: #ccsd(Y1|X1,Y2|X2)
    m=len(Xp)
    kxx = calculate_gram(torch.cat([Xp, Xq],0), sigma=sigmas[0])[0] 
    kyy = calculate_gram(torch.cat([Yp, Yq],0), sigma=sigmas[1])[0] 
    kxpp = kxx[:m,:m]  
    kxqq = kxx[m:,m:]
    kxpq = kxx[:m,m:] 
    kxqp = kxx[m:,:m] 
    div_x_pq = kxpp.mean().log2() + kxqq.mean().log2() - kxpq.mean().log2() - kxqp.mean().log2()
    
    kypp = kyy[:m,:m] 
    kyqq = kyy[m:,m:]  
    kypq = kyy[:m,m:] # m*n
    kyqp = kyy[m:,:m] # n*m

    power_p = atleast_epsilon(atleast_epsilon(kxpp*kypp).sum(1)/atleast_epsilon(kxpp.sum(1)**2)).sum().log2()
    power_q = atleast_epsilon(atleast_epsilon(kxqq*kyqq).sum(1)/atleast_epsilon(kxqq.sum(1)**2)).sum().log2()
    cross_p = atleast_epsilon(atleast_epsilon(kxpq*kypq).sum(1)/atleast_epsilon(kxpp.sum(1)*kypq.sum(1))).sum().log2()
    cross_q = atleast_epsilon(atleast_epsilon(kxqp*kyqp).sum(0)/atleast_epsilon(kxqq.sum(1)*kxqp.sum(0))).sum().log2()    
    div_cyx_pq=power_p+power_q-cross_p-cross_q

    return (div_x_pq, div_cyx_pq)

def gcsd(*domains:Union[list,tuple,Tensor], 
         sigmas: Union[Tensor,ndarray,float]=None,
         **kwargs) -> Tensor:
    dn = len(domains)
    if sigmas is None:
        x=torch.cat(domains,0)
        # sigmas = kernel_size_estimation(p_dist_2(x,x)) 
        sigmas = get_kernel_size(x)
    # domain gram_mat
    # gram_mat = calculate_gram_comprehensive(domains, sigmas, **kwargs)[0]
    gram_mat = calculate_gram_transversal(*domains, sigmas=sigmas, **kwargs)[0]
    # GCSD-Inter-Domain discrepancy
    tri_d = torch.arange(dn)

    Gs=gram_mat.mean(-1)
    Gp=Gs[tri_d,tri_d,...]
    Gc=Gs.prod(1)/Gp

    cross_entropy=-atleast_epsilon( Gc,       ).mean().log()    
    power_entropy=-atleast_epsilon((Gp**(dn-1)).mean(-1)).log().mean()
    if torch.isnan(cross_entropy) or torch.isnan(power_entropy):
        print('exists NAN in gcsd',(cross_entropy - power_entropy)/dn) 
    return (cross_entropy - power_entropy), sigmas

def gcsd_QMI(*variants: Union[list,tuple,Tensor], 
        sigmas: Union[Tensor,ndarray,float]=None,
        **kwargs) -> Tensor:
    dn = len(variants)
    tri_d = torch.arange(dn)
    All, sigmas_est = calculate_gram_transversal(*variants, sigmas=sigmas)
    Gp = All[tri_d,tri_d,...]
    if sigmas is None:
        sigmas = sigmas_est[tri_d,tri_d]

    P_joint=Gp.prod(0).mean(-1)    
    P_marginal=Gp.mean(-1)

    cross = -2.0*atleast_epsilon(All.mean(-2).prod(0).mean()).log()
    power_joint  = atleast_epsilon(P_joint.mean()).log()
    power_product= atleast_epsilon(P_marginal.mean(-1)).log().sum()

    multi_var_CSD_qmi = cross + power_joint + power_product

    return multi_var_CSD_qmi, sigmas

def i_gcsd_QMI(*variants: Union[list,tuple,Tensor], 
        sigmas: Union[Tensor,ndarray,float]=None,
        **kwargs) -> Tensor:
    dn = len(variants)
    tri_d = torch.arange(dn)
    All, sigmas_est = calculate_gram_transversal(*variants, sigmas=sigmas)
    All = All + 1.0## to ensure precision protection of product operation
    Gp = All[tri_d,tri_d,...]
    if sigmas is None:
        sigmas = sigmas_est[tri_d,tri_d]

    P_joint=Gp.prod(0).mean(-1)    
    P_marginal=Gp.mean(-1)

    cross = -2.0*atleast_epsilon(All.mean(-2).prod(0).mean()).log()
    power_joint  = atleast_epsilon(P_joint.mean()).log()
    power_product= atleast_epsilon(P_marginal.mean(-1)).log().sum()

    multi_var_CSD_qmi = cross + power_joint + power_product

    return multi_var_CSD_qmi, sigmas

def gccsd(domains_y:Union[list,tuple,Tensor], 
         domains_z:Union[list,tuple,Tensor],
         sigmas_y: Union[Tensor,ndarray,float]=None,
         sigmas_z: Union[Tensor,ndarray,float]=None,
         **kwargs) -> Tensor:
    dn, bn = len(domains_y), domains_y[0].shape[-2]
    tri_d = torch.arange(dn)
    # domain gram_mat
    Kyy = calculate_gram_transversal(*domains_y, sigmas=sigmas_y, **kwargs)[0]
    Kzz = calculate_gram_transversal(*domains_z, sigmas=sigmas_z, **kwargs)[0]
    # GCSD-Inter-Domain discrepancy
    Kp=atleast_epsilon(Kyy*Kzz).sum(-1)
    Gkp=Kp[tri_d,tri_d,...]

    Kzz_p = atleast_epsilon(Kzz.sum(-1))
    Gzp   = atleast_epsilon(Kzz[tri_d,tri_d,...].sum(-1))

    cross_entropy_c = -atleast_epsilon((atleast_epsilon(Kp.prod(1)/Gkp)/atleast_epsilon(Kzz_p.prod(1)))).mean().log()
    power_entropy_c = -atleast_epsilon(atleast_epsilon(atleast_epsilon(Gkp/Gzp)**(dn-1 ))/Gzp).sum(-1).log().mean()

    
    cross_entropy_z = -atleast_epsilon(Kzz_p.prod(1)).mean().log()    
    power_entropy_z = -atleast_epsilon(((Gzp)**(dn-1))).sum(-1).log().mean()

    # if torch.isnan(cross_entropy_z) or torch.isnan(power_entropy_z):
    #     print('exists NAN in gcsd',(cross_entropy_z - power_entropy_z)/dn) 
    return (cross_entropy_z - power_entropy_z)/dn, (cross_entropy_c - power_entropy_c)/dn

def pcsd(*groups:Union[list,tuple], **kwargs) -> Tensor:
    return average_pairwise_div(groups, div_func=csd, **kwargs)

def gcsd_cluster(*clusters:Union[list,tuple],
                 assignment:Union[ndarray,Tensor]=None,
                 sigma: Union[Tensor,ndarray,float]=None,
                 **kwargs) -> Tensor:
    G = calculate_gram(*clusters, sigma=sigma, **kwargs)[0]
    A = assignment
    m = A.shape[1]
    k = m-1
    Ak = atleast_epsilon(A**k, EPS)
    Ga = atleast_epsilon(G.matmul(A), EPS)
    Gak = atleast_epsilon(Ga**k, EPS)   
    Gap = atleast_epsilon(torch.cat([torch.cat([Ga[:,:j], Ga[:,j+1:]],1).prod(1,True) for j in range(m)],1), EPS)
    cross_entropy = -atleast_epsilon((Ak*Gap).sum()/m, EPS).log() # - m*torch.log(torch.tensor(n))
    power_entropy = -atleast_epsilon((Ak.T.matmul(Gak)).diag(), EPS).log().mean() # - m*torch.log(torch.tensor(n))    
    return torch.exp(-(cross_entropy - power_entropy)/k)

def csd_pdf(X:Tensor, Y:Tensor, Px:T_dists, Py:T_dists)->Tensor:
    PX = Px.log_prob(X).exp()
    QX = Py.log_prob(X).exp()
    QY = Py.log_prob(Y).exp()
    PY = Px.log_prob(Y).exp()
    cross = -atleast_epsilon(0.5*(QX.mean()+PY.mean())).log()
    power = -0.5*(atleast_epsilon((PX).mean()).log()+
                  atleast_epsilon((QY).mean()).log())

    return cross - power

def pcsd_pdfs(*groups:Union[list,tuple], dists:T_dists=None, **kwargs) -> Tensor:
    return average_pairwise_div_pdfs(groups, dists, div_func=csd_pdf, **kwargs)

def gcsd_pdfs(*groups:Union[list,tuple], dists:T_dists=None, **kwargs) -> Tensor:      
    """
    math:
    \begin{array}{c} 
    {D_{GCS}} \approx  
        - \log \sum\limits_i^n {\prod\limits_t^m {{p_t}\left( {{x_i}} \right)} }  \\
        + \frac{1}{m}\sum\limits_t^m {\log \sum\limits_i^n {p_t^m\left( {{x_i}} \right)} } 
    \end{array}
    """   
    PDF = torch.stack([torch.stack([dist.log_prob(group).exp() for group in groups]) for dist in dists]) # C*C*N
    C = PDF.shape[0]
    tr_index = torch.arange(C)

    cross = -atleast_epsilon((PDF.prod(0)/PDF[tr_index,tr_index,:]).mean()).log()
    power = -atleast_epsilon(((PDF[tr_index,tr_index,:]**(C-1)).mean(-1))).log().mean()

    return (cross-power)

############ jrd #######################
def jrd(X:Tensor, Y:Tensor, alpha:float=2.0, **kwargs)->Tensor:
    """fusion with pdf
    """
    k=torch.tensor(alpha-1,device=X.device)
    if 'weights' in kwargs:
        B = Tensor(kwargs['weights']).to(X.device)       
    else:
        B = torch.tensor([0.5,0.5]).to(X.device) 
    m=len(X)
    k = calculate_gram(torch.cat([X, Y],0))[0]     
    p11 = k[:m,:m].mean(-1) #etismate p1 with sampleset1
    p22 = k[m:,m:].mean(-1) #etismate p2 with sampleset2
    p12 = k[:m,m:].mean(-1) #etismate p1 with sampleset2
    p21 = k[m:,:m].mean(-1) #etismate p2 with sampleset1
    p_cross =  ((B[0]*p11+B[1]*p12)**k + (B[0]*p21+B[1]*p22)**k)/2
        
    cross = -atleast_epsilon(p_cross.mean()).log()
    power = -(atleast_epsilon((p11**k).mean()).log()*B[0]+
              atleast_epsilon((p22**k).mean()).log()*B[1])
    
    return (cross - power)/k# + torch.tensor(2, device=cross.device).log()

def pjrd(*groups:Union[list,tuple], **kwargs) -> Tensor:
    return average_pairwise_div(groups, div_func=jrd, **kwargs)

def gjrd(*domains:Union[list,tuple,Tensor], 
         sigmas: Union[Tensor,ndarray,float]=None,
         **kwargs) -> Tensor:
    dn = len(domains)
    if 'weights' in kwargs:
        B = Tensor(kwargs['weights']).to(domains[0].device)
        if len(B)==2 and dn>2:
            B = Tensor([B[0]/(dn-1)]*(dn-1)+[B[1]]).to(domains[0].device)/B.sum()        
    else:
        B = torch.ones([dn,]).to(domains[0].device)/dn
    # gram_mat = calculate_gram_comprehensive(domains, sigmas, **kwargs)[0]
    gram_mat = calculate_gram_transversal(*domains, sigmas=sigmas, **kwargs)[0]
    # GJRD-Inter-Domain discrepancy
    if 'order' in kwargs:
        order = kwargs.get('order')
    elif 'params' in kwargs:
        order = kwargs.get('params')
    elif 'alpha' in kwargs:
        order = kwargs['alpha']
    else:
        order = 2    
    
    k=torch.tensor(order-1).to(domains[0].device)
    
    tri_d = torch.arange(dn)

    PDF = gram_mat.mean(-1)
    G1 = (B.view(1,-1).matmul(PDF))**k    
    G2 = (PDF[tri_d,tri_d,:])**k

    cross_entropy=-atleast_epsilon(G1.mean()).log()
    power_entropy=-atleast_epsilon(G2.mean(-1)).log().matmul(B)

    return (cross_entropy - power_entropy)/k

def igjrd(*domains:Union[list,tuple,Tensor], 
         sigmas: Union[Tensor,ndarray,float]=None,
         **kwargs) -> Tensor:
    # GJRD-Inter-Domain discrepancy
    dn = len(domains)
    tri_d = torch.arange(dn)
    if 'weights' in kwargs:
        B = Tensor(kwargs['weights']).to(domains[0].device)
        if len(B)==2 and dn>2:
            B = Tensor([B[0]/(dn-1)]*(dn-1)+[B[1]]).to(domains[0].device)/B.sum()
    else:
        B = torch.ones([dn,]).to(domains[0].device)/dn
    if 'order' in kwargs:
        order = kwargs.get('order')
    elif 'params' in kwargs:
        order = kwargs.get('params')
    elif 'alpha' in kwargs:
        order = kwargs['alpha']
    else:
        order = 2    
    k=torch.tensor(order-1).to(domains[0].device)

    # gram_mat = calculate_gram_comprehensive(domains, sigmas, **kwargs)[0]
    domain_expand = sum([beta*domain for beta,domain in zip(B,domains)])
    domain_expand = [domain_expand] if isinstance(domains, list) else (domain_expand,) + domains
    gram_mat_expand = calculate_gram_transversal(*domain_expand, sigmas=sigmas, **kwargs)[0]
    cross_mat = (gram_mat_expand[0,1:,...].mean(-1))**k
    pover_mat = (gram_mat_expand[1:,1:,...][tri_d,tri_d,...].mean(-1))**k
        
    cross_entropy=-atleast_epsilon(B.view(1,-1).matmul(cross_mat).mean()).log()
    power_entropy=-atleast_epsilon(pover_mat.mean(-1)).log().matmul(B)

    return (cross_entropy - power_entropy)/k

def gjrd_cluster(*clusters:Union[list,tuple],
                 assignment:Union[ndarray,Tensor]=None,
                 sigma: Union[Tensor,ndarray,float]=None, 
                 order:float=2.0,
                 **kwargs) -> Tensor:
    G = calculate_gram(*clusters, sigma=sigma, **kwargs)[0]
    A = assignment
    n, m = tuple(A.shape)
    k = torch.tensor([order-1]).to(clusters[0].device)
    if k==1: # special case of quadratic divergence
        AGA = atleast_epsilon(A.T.matmul(G).matmul(A)/n**2, eps=EPS)
        cross_entropy = -AGA.mean().log()
        power_entropy = -AGA.diag().log().mean()

    else:
        AkT = atleast_epsilon(A**k, EPS).T
        Gak = atleast_epsilon((G.matmul(A))**k, EPS)        

        cross_entropy = -atleast_epsilon(((G.sum(1)/m)**k).sum()/m, EPS).log()
        power_entropy = -atleast_epsilon(AkT.matmul(Gak).diag(), EPS).log().mean()
    return torch.exp(-(cross_entropy-power_entropy)/k)

def jrd_pdf(X:Tensor, Y:Tensor, Px:T_dists, Py:T_dists, alpha:float=2.0)->Tensor:
    k = torch.tensor([alpha-1]).to(X.device)
    PX = Px.log_prob(X).exp()
    QX = Py.log_prob(X).exp()
    QY = Py.log_prob(Y).exp()
    PY = Px.log_prob(Y).exp()
    
    cross = -atleast_epsilon(((0.5*(PX+QX))**k+(0.5*(PY+QY))**k).mean()*0.5).log()
    power = -(atleast_epsilon((PX**k).mean()).log()+atleast_epsilon((QY**k).mean()).log())*0.5

    return (cross-power)/(k)

def gjrd_pdfs(*groups:Union[list,tuple], dists:T_dists=None, alpha:float=2.0, **kwargs) -> Tensor:   
    """
    math:
    \begin{array}{c}
        D_{{\rm{GJR}}}^\alpha  = \frac{1}{{\alpha  - 1}}\left( { - \log {{\int\limits_{\cal X} {\left[ {\sum\limits_t {{\beta _t}{p_t}\left( x \right)} } \right]} }^\alpha }dx + \sum\limits_{t = 1}^m {{\beta _t}\log \int\limits_{\cal X} {p_t^\alpha \left( x \right)dx} } } \right)\\
        \approx \frac{1}{{\alpha  - 1}}\left( { - \log \sum\limits_i^n {{{\left[ {\sum\limits_t {{\beta _t}{p_t}\left( {{x_i}} \right)} } \right]}^\alpha }}  + \sum\limits_{t = 1}^m {{\beta _t} \log \sum\limits_i^n { p_t^\alpha \left( {{x_i}} \right)} } } \right)
    \end{array}
    """
    PDF = torch.stack([torch.stack([dist.log_prob(group).exp() for group in groups]) for dist in dists]) # C*C*N
    C = PDF.shape[0]
    tr_index = torch.arange(C)
    k = torch.tensor([alpha-1]).to(groups[0].device)
    weights = kwargs.get('weights')
    if not (weights is not None and len(weights)==C and weights.sum()==1):
        weights =  torch.ones([C, 1, 1], device=PDF.device, dtype=PDF.dtype)/C

    cross = -atleast_epsilon(((((weights*PDF).sum(0))**k).mean())).log()
    power = -(atleast_epsilon((PDF[tr_index,tr_index,:]**k).mean(-1)).log()*(weights.squeeze())).sum()

    return (cross-power)/k

def pjrd_pdfs(*groups:Union[list,tuple], dists:T_dists=None, **kwargs) -> Tensor:
    return average_pairwise_div_pdfs(groups, dists, div_func=jrd_pdf, **kwargs)

##########################  multi-MMD #######################
def mmd(X:Tensor, Y:Tensor, sigma=None, **kwargs):
    m = X.shape[0]
    n = Y.shape[0]
    k = calculate_gram(torch.cat([X, Y],0), sigma=sigma)[0]     

    kxx, kyy, kxy = k[:m,:m], k[m:,m:], k[:m,m:]

    d  = (kxx.sum() - kxx.trace())/(m*(m-1))
    d += (kyy.sum() - kyy.trace())/(n*(n-1))
    d -=  kxy.sum()*2/(n*m) + torch.tensor(1e-6, device=X.device)

    # if d.is_cuda:
    #     d = d.cpu()
    return d

def mmd_multi_kernel(X:Tensor, Y:Tensor, sigma=None, **kwargs):
    return None

def gmmd(*groups:Union[list,tuple], sigmas=None, **kwargs) -> Tensor:
    gram_mat_expand = calculate_gram_transversal(*groups, sigmas=sigmas, **kwargs)[0]
    n_group, n_sample = gram_mat_expand.shape[0], gram_mat_expand.shape[-1]
    diag_sample=torch.arange(n_sample,n_sample,1)
    diag_group=torch.arange(n_group)
    triu_group=torch.triu_indices(n_group,n_group,1)
    
    power_mat = gram_mat_expand[diag_group,diag_group,...]
    cross_mat = gram_mat_expand[triu_group[0],triu_group[1],...]
    power = (power_mat.sum([-1,-2])-power_mat[:,diag_sample,diag_sample].sum(-1)).sum()/(n_sample**2-n_sample)
    cross = cross_mat.mean(-1).mean(-1)

    gmmd = (power - cross.sum()/n_group)/(n_group*2)

    return gmmd

def pmmd(*groups:Union[list,tuple],  **kwargs) -> Tensor:    
    return average_pairwise_div(groups, div_func=mmd, **kwargs)

def pkld(*groups:Union[list,tuple],  **kwargs) -> Tensor:    
    return average_pairwise_div(groups, div_func=kld, **kwargs)

#####################################################
def average_pairwise_div(groups:Union[list,tuple], sigmas=None, div_func=None, use_eec:bool=True, **kwargs) -> Tensor: 
    n_group=len(groups)
    if div_func in globals():
        pd=[div_func(groups[i],groups[j],sigma=None if sigmas is None else sigmas[i,j], 
                        **kwargs) for i in range(n_group) for j in range(i)]
    else:
        pd=[div_func(groups[i],groups[j]) for i in range(n_group) for j in range(i)]
    
    len_pd=len(pd)
    comb_pd=len_pd*(len_pd-1)/2.0
    if len_pd>1 and use_eec:
        pdiv = (sum([torch.nn.functional.mse_loss(pd[i],pd[j]) for i in range(len_pd) for j in range(i)])/comb_pd).sqrt()
    else:
        pdiv = sum(pd)/len_pd
    return pdiv, 1.0
    
def average_pairwise_div_pdfs(groups:Union[list,tuple], dists:T_dists=None, div_func=None, **kwargs) -> Tensor: 
    n_group=len(groups)
    pd=[div_func(groups[i],groups[j],dists[i],dists[j],**kwargs 
                     ) for i in range(n_group) for j in range(i)]
    return sum(pd)*2.0/(n_group**2-n_group)

def bary_center_dissimilarity(*dists, bary_center=None, measure='mmd', weights=[1.0], use_eec:bool=True, **kwargs):
    n=len(dists)
    if len(weights)!=n:
        if len(weights)==2:
            weights=[weights[0]/(n-1)]*(n-1)+[weights[1]]
        else:
            weights=[1.0/n]*n
    if not bary_center:
        bary_center = dists[-1] 
    measure = measure if callable(measure) else globals()[measure]
    divs = torch.stack([measure(dist, bary_center) for dist in dists])
    # eec = mee(divs)[0]
    eec = average_pairwise_div(divs,div_func=torch.nn.functional.mse_loss)[0].sqrt()
    weights_ = torch.tensor(weights).to(divs.device)
    bcd = (divs*weights_).sum()
    return (eec if use_eec else bcd, divs)
    # return 

def conditional_bary_center_dissimilarity(Z, Y, measure='csd', weights=[1.0], 
                                          sigma_z=None, sigma_y=None, **kwargs):
    
    sigma_z = get_kernel_size(torch.cat(Z))
    sigma_y = get_kernel_size(torch.cat(Y))

    y_bc = Y.pop(-1)
    z_bc = Z.pop(-1)
    n=len(Y)
    if len(weights)!=n:
        if len(weights)==2:
            weights=[weights[0]/(n-1)]*(n-1)+[weights[1]]
        else:
            weights=[1.0]*n

    c_measure = globals()['c'+(measure.__name__ if callable(measure) else measure)]
    all_divs = [c_measure(z, z_bc, y, y_bc, [sigma_z, sigma_y]) for (z,y) in zip(Z,Y)]
    divs = torch.stack([div[0] for div in all_divs])
    cdivs = torch.stack([div[1] for div in all_divs])
    weights_ = torch.tensor(weights).to(divs.device)
    bcd = (divs*weights_).sum()
    cbcd = (cdivs*weights_).sum()
    return ((bcd, divs),(cbcd, cdivs))
########################################################
def gcsd_cluster(assignment, feature):
    A, G = assignment, calculate_gram(feature)[0]
    m = assignment.shape[1]
    
    k=m-1
    Ak = atleast_epsilon(A**k, EPS)
    Ga = atleast_epsilon(G.matmul(A), EPS)
    Gak = atleast_epsilon(Ga**k, EPS)   

    # # version_1
    # Ap = row_prod_except(A)
    # cross_entropy = -atleast_epsilon((Ga*Ap).sum()/m, EPS).log() # - m*torch.log(torch.tensor(n))
    # power_entropy = -atleast_epsilon((Ak.T.matmul(Gak)).diag(), EPS).log().mean() # - m*torch.log(torch.tensor(n))
    
    ## version_2
    Gap = row_prod_except(Ga)
    cross_entropy = -atleast_epsilon((Ak*Gap).sum()/m, EPS).log() # - m*torch.log(torch.tensor(n))
    power_entropy = -atleast_epsilon((Ak.T.matmul(Gak)).diag(), EPS).log().mean() # - m*torch.log(torch.tensor(n))
    
    return torch.exp(-(cross_entropy - power_entropy)/k)

def gjrd_cluster(assignment, feature, order=2):
    A, G = assignment, calculate_gram(feature)[0]
    n, m = tuple(A.shape)
    k = order-1
    if k==1:
        AGA = atleast_epsilon(A.T.matmul(G).matmul(A)/n**2, eps=EPSILON)
        cross_entropy = -AGA.mean().log()
        power_entropy = -AGA.diag().log().mean()

    else:
        AkT = atleast_epsilon(A**k, EPS).T
        Gak = atleast_epsilon((G.matmul(A))**k, EPS)        

        cross_entropy = -atleast_epsilon(((G.sum(1)/m)**k).sum()/m, EPS).log()
        power_entropy = -atleast_epsilon(AkT.matmul(Gak).diag(), EPS).log().mean()

    return torch.exp(-(cross_entropy-power_entropy)/k)
    # return torch.exp(-(cross_entropy-power_entropy))

def bce_unsupervised(preds, bounds=[0.6,0.4], eps:float=1e-06):
    # loss = -r_mask*log(r_orig)-(1-r_mask)*log(1-r_orig)
    triu_indices = torch.triu_indices(len(preds),len(preds),1)
    norm = preds.norm(dim=1).unsqueeze(0).repeat(len(preds),1)
    sim_orig = (preds.matmul(preds.T)/(norm*norm.T))[triu_indices[0],triu_indices[1]]    
    # sim_orig=torch.cat([torch.nn.functional.cosine_similarity(preds[i:i+1],preds[j:j+1]) for i in range(len(preds)) for j in range(i+1,len(preds))])


    sim_orig = torch.masked_select(sim_orig, (sim_orig>bounds[0]) | (sim_orig<bounds[1]))
    if len(sim_orig)>0:
        sim_mask = torch.where(sim_orig>bounds[0], torch.ones_like(sim_orig), sim_orig)
        sim_mask = torch.where(sim_mask<bounds[1], torch.zeros_like(sim_mask), sim_mask)
        bce = -sim_mask*sim_orig.clamp(eps,1.0).log() - (1-sim_mask)*(1-sim_orig).clamp(eps,1.0).log()
    else:
        bce = None

    return None if bce is None else bce.mean()

def keep_corner_simplex(
                 assignment:Union[ndarray,Tensor],
                 clusters:Union[list,tuple,Tensor]=None,
                 div_func:callable=None,
                 **kwargs)->Tensor:
    # minimize distance (A[:,i], eye(j,:)) to obtain simplex A
    A = assignment
    n,m=tuple(A.shape)
    # E = torch.eye(A.shape[1]).type_as(A)
    # Q = ((A.unsqueeze(-1).repeat(1,1,m)-E.unsqueeze(0).repeat(n,1,1))**2).sum(-1)
    # Q = torch.exp(-Q).softmax(1)
    # if div_func is not None:
    #     return div_func(clusters, Q, **kwargs)
    # else: 
    #     M = Q.mm(Q.T)
    #     return 1.0 + M.triu(1).sum()*2/(n**2-n)-M.diag().mean()
    #     # return M.triu(1).sum()*2/(n**2-n)/(M.diag().mean()+1.0e-10)

    simplex=1.0-(A**2).sum(1).mean() # to cornered-simplex
    diff = -p_dist_2(A,A).mean() # to sample-different

    return simplex + diff/(2.*m)

def keep_orthogonal(assignment:Union[ndarray,Tensor])->Tensor:
    # maximize trace(AA') to achieve onehot prediction
    # minimize triu(AA') simultaneously will speed up the convergency
    M = assignment.mm(assignment.T)
    n=M.shape[0]
    return 1.0 + M.triu(1).sum()*2/(n**2-n)-M.diag().mean()
    # return (M.triu(1).sum()*2/(n**2-n))/(M.diag().mean()+1.0e-10)



def pmse(*groups:Union[list,tuple], **kwargs) -> Tensor:
    return average_pairwise_div(groups, div_func=torch.nn.functional.mse_loss, **kwargs)
##########################################################
Loss = {
    'ent': renyi_entropy,
	'joint_ent': joint_entropy,
    'mi':  mutual_information,
    'kld': kld,
    'mmd': mmd,
    'jrd': jrd,
    'csd': csd,
    'gcsd': gcsd,
    'pcsd': pcsd,
    'gjrd': gjrd,
    'pjrd': pjrd,
    'pmmd': pmmd,
    'pkld': pkld,
    'pmse': pmse,
    'bcd': bary_center_dissimilarity,
    'gcsd_cluster': gcsd_cluster,
    'gjrd_cluster': gjrd_cluster,
    'ccsd': ccsd, 
    'gccsd': gccsd,
    'csd_qmi': csd_QMI,
    'gcsd_qmi': gcsd_QMI,
    'i_gcsd_qmi': i_gcsd_QMI,
    'cbcd': conditional_bary_center_dissimilarity,
    'ce': torch.nn.functional.cross_entropy,
    'orth': keep_orthogonal,
    'simplex':keep_corner_simplex,
    'contrastive': bce_unsupervised, 
    }