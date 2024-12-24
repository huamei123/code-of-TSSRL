import torch
from torch import nn
from models import SAINT, SAINT_vision

from data_openml import data_prep_openml,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise
from itertools import chain
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt_sne


parser.add_argument('--dset_id', required=True, type=int)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', required=True, type=str,choices = ['binary','multiclass','regression'])
# parser.add_argument('--task', default='multiclass', type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.3, type=float)
parser.add_argument('--ff_dropout', default=0.05, type=float)
# parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])
parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='SGD', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0003, type=float)
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--batchsize', default=1024, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default=1, type=int)
parser.add_argument('--dset_seed', default=1, type=int)
parser.add_argument('--active_log', action='store_true')

parser.add_argument('--pretrain', action = 'store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices=['contrastive','contrastive_sim','denoising'])
# parser.add_argument('--pt_tasks', default=['denoising'], type=str,nargs='*',choices=['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=['mixup','cutmix'], type=str,nargs='*',choices=['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.2, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_noise_type', default='missing', type=str, choices=['missing','cutmix'])
parser.add_argument('--train_noise_level', default=0, type=float)

parser.add_argument('--ssl_samples', default=5, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.99, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices=['common','sep'])

# dset_id = 188

opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb
    if opt.train_noise_type is not None and opt.train_noise_level > 0:
        wandb.init(project="saint_v2_robustness", group =f'{opt.run_name}_{opt.task}' ,name = f'{opt.task}_{opt.train_noise_type}_{str(opt.train_noise_level)}_{str(opt.attentiontype)}_{str(opt.dset_id)}')
    elif opt.ssl_samples is not None:
        wandb.init(project="saint_v2_ssl", group = f'{opt.run_name}_{opt.task}' ,name = f'{opt.task}_{str(opt.ssl_samples)}_{str(opt.attentiontype)}_{str(opt.dset_id)}')
    else:
        raise'wrong config.check the file you are running'
    wandb.config.update(opt)


print('Downloading and processing the dataset, it might take some time.')
cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean, train_std = data_prep_openml(opt.dset_id, opt.dset_seed,opt.task, datasplit=[.7, .3])
continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32)

##### Setting some hyperparams based on inputs and dataset
_,nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(4,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 2
    opt.attention_heads = 8
    opt.attention_dropout = 0.8
    opt.embedding_size = 16
    if opt.optimizer =='SGD':
        opt.ff_dropout = 0.25
        opt.lr = 0.000105
    else:
        opt.ff_dropout = 0.8

if opt.dset_id in [41540, 42729, 42728]:
    opt.batchsize = 2048

if opt.dset_id == 42734 :
    opt.batchsize = 255
print(nfeat,opt.batchsize)
print(opt)



train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=0)



test_ds = DataSetCatCon(X_test, y_test, cat_idxs,opt.dtask, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=0)
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:,0]))

cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.



model = SAINT(
categories = tuple(cat_dims),
num_continuous = len(con_idxs),
dim = opt.embedding_size,
dim_out = 1,
depth = opt.transformer_depth,
heads = opt.attention_heads,
attn_dropout = opt.attention_dropout,
ff_dropout = opt.ff_dropout,
mlp_hidden_mults = (4, 2),
cont_embeddings = opt.cont_embeddings,
attentiontype = opt.attentiontype,
final_mlp_style = opt.final_mlp_style,
y_dim = y_dim
)
vision_dset = opt.vision_dset

if y_dim == 2 and opt.task == 'binary':
    # opt.task = 'binary'
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and  opt.task == 'multiclass':
    # opt.task = 'multiclass'
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise'case not written yet'


model.to(device)


if opt.pretrain:
    from pretraining import TSSRL_pretrain
    model = TSSRL_pretrain(model, cat_idxs,X_train,y_train, continuous_mean_std, opt,device)

if opt.ssl_samples is not None and opt.ssl_samples > 0 :
    print('We are in semi-supervised learning case')
    X = []
    X_m = []
    y = []
    # y_train = list(chain.from_iterable(y_train['data']))
    for i in range(15):
        indices = np.where(y_train['data']==[i])
        tempx = X_train['data'][indices[0],:][:opt.ssl_samples]
        tempx_m = X_train['mask'][indices[0]][:opt.ssl_samples]
        tempy = (np.ones([opt.ssl_samples,1])*i).astype(int)
        X.append(tempx)
        y.append(tempy)
        X_m.append(tempx_m)
    X = np.concatenate(X)
    X_m = np.concatenate(X_m)
    y = np.concatenate(y)
    y_train = {
        'data': y.reshape(-1, 1)
    }
    X_train = {
        'data': X,
        'mask': X_m
    }

    train_bsize = min(opt.ssl_samples*1,opt.batchsize)

    train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=train_bsize, shuffle=True,num_workers=0)


if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000
print('Training begins now.')
# if __name__=='__main__':
for epoch in range(opt.epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        model.train()
        optimizer.zero_grad()
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), \
            data[3].to(device), data[4].to(device)
        if opt.train_noise_type is not None and opt.train_noise_level > 0:
            noise_dict = {
                'noise_type': opt.train_noise_type,
                'lambda': opt.train_noise_level
            }
            if opt.train_noise_type == 'cutmix':
                x_categ, x_cont = add_noise(x_categ, x_cont, noise_params=noise_dict)
            elif opt.train_noise_type == 'missing':
                cat_mask, con_mask = add_noise(cat_mask, con_mask, noise_params=noise_dict)

        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
        reps = model.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:, 0, :]
        y_outs = model.mlpfory(y_reps)
        if opt.task == 'regression':
            loss = criterion(y_outs, y_gts)
        else:
            loss = criterion(y_outs, y_gts.reshape(-1).long())
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    # print(running_loss)
    if opt.active_log:
        wandb.log({'epoch': epoch, 'train_epoch_loss': running_loss,
                   'loss': loss.item()
                   })
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            if opt.task in ['binary', 'multiclass']:
                test_accuracy, test_auroc, p_true, p_pre, _ = classification_scores(model, testloader, device, opt.task, vision_dset)
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                      (epoch + 1, test_accuracy, test_auroc))
                if opt.active_log:
                    # wandb.log({'valid_accuracy': accuracy, 'valid_auroc': auroc})
                    wandb.log({'test_accuracy': test_accuracy, 'test_auroc': test_auroc})
                if opt.task == 'multiclass':
                    if test_accuracy > best_valid_accuracy:
                        best_valid_accuracy = test_accuracy
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(), '%s/bestmodel.pth' % (modelsave_path))
                else:
                    if test_auroc > best_valid_auroc:
                        best_valid_auroc = test_auroc
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(), '%s/bestmodel.pth' % (modelsave_path))
        model.train()







