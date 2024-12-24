import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset



def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2]):
    
    np.random.seed(seed)
    data = pd.read_csv("C:/Users\HNU\Desktop\自监督学习/new_data\SZCAV.csv")
    data = data.sample(n=len(data["Labels"]), random_state=13, axis=0)
    data = data.reset_index(drop=True)
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    categorical_columns = []
    categorical_indicator = [False] * 11
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "test"], p=datasplit, size=(X.shape[0],))

    train_indices = X[X.Set == "train"].index
    # valid_indices = X[X.Set == "valid"].index
    test_indices = X[X.Set == "test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)#除缺失值之外的数值都设为int
    
    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))

    for col in cont_columns:
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)#用均值填充缺失值

    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    X_train, y_train = data_split(X,y,nan_mask,train_indices)

    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)#计算每一列的均值和方差
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean, train_std




class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

