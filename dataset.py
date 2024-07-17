from tkinter import X
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset

EPS = 1e-16


def get_train_test_splits(paths, train_components=(1, 2, 3), test_components=(6, 7, 8)):
    df = pd.concat([pd.read_csv(p) for p in paths]).reset_index()
    n_comp = max(int(x[10:]) for x in df.columns if "Box 0 mol" in x) + 1
    df["Components"] = (df[[f"Init mol {i}" for i in range(n_comp)]] > 0).sum(axis=1)
    df_train = df[np.isin(df["Components"], train_components)]
    df_test = df[np.isin(df["Components"], test_components)]
    return df_train, df_test, n_comp


def preprocess(df, ncomp, y_norm, mask_value=-1, use_partial_pressure=True):
    temp = 1000 / df["Temperature"].values
    pres = df["Box 1 pressure"].values
    ninit = df[["Init mol %d" % x for x in range(ncomp)]].values
    nvap = df[["Box 1 mol %d" % x for x in range(ncomp)]].values
    
    Y = df[["Box 0 mol %d" % x for x in range(ncomp)]].values / y_norm
    if use_partial_pressure:
        pres_part = pres.reshape(-1, 1) * nvap / np.sum(nvap, axis=1).reshape(-1, 1)
        feature = pres_part
    else:
        molfrac = ninit / np.sum(ninit, axis=1).reshape(-1, 1)
        feature = molfrac
    X = np.concatenate([feature.reshape(-1, ncomp, 1), 
                            np.tile(temp.reshape(-1, 1, 1), [1, ncomp, 1]),
                            np.tile(pres.reshape(-1, 1, 1), [1, ncomp, 1]),
                            ], axis=2)
    X[:, :, 0] = np.log(X[:, :, 0] + EPS)
    X[:, :, 2] = np.log(X[:, :, 2] + EPS)
    X[ninit == 0] = mask_value
    return torch.tensor(X).float(), torch.tensor(Y).float()


def make_datasets(df_train, df_test, n_comp, mask_value=-1, return_norm=False):
    loading_norm = np.max(df_train[["Box 0 mol %d" % x for x in range(n_comp)]].values, axis=0)
    X_train, Y_train = preprocess(df_train, n_comp, loading_norm, mask_value=mask_value)
    X_test, Y_test = preprocess(df_test, n_comp, loading_norm, mask_value=mask_value)
    data_train = TensorDataset(X_train, Y_train)
    data_test = TensorDataset(X_test, Y_test)
    if return_norm:
        return data_train, data_test, loading_norm
    else:    
        return data_train, data_test  

if __name__ == "__main__":
    df_train, df_test, n_comp = get_train_test_splits(
        ["data-binary-ternary/MFI-0.csv", "data-full/MFI-0.csv"],
    )
    data_train, data_test = make_datasets(df_train, df_test, n_comp)