# Standard libraries
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import notebook as tqdm_notebook
import seaborn as sns
from IPython.display import Markdown, display
import json
from datetime import datetime
import random

# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn

# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

import importlib






def load_data(path, filter_yield=False, filter_wheat=False, filter_years=False, df=None, correlation_filters=None):

    if df is None:
        print("loading df")
        df = pd.read_feather(path)
    if correlation_filters is None:
        print("loading correlation_filters")
        f = open(path.replace(".feather", ".json"), "r")
        correlation_filters = json.load(f)
        f.close()

    if 'level_0' in df.columns:
        df = df.drop(columns=['level_0'])

    pmap = np.array(list(set(df['polygon_id'])))
    polygons = [np.where(np.array(pmap) == i)[0][0] for i in df['polygon_id']]
    df['polygon_id'] = polygons

    df = df.reset_index()
    if 'level_0' in df.columns:
        df = df.drop(columns=['level_0', 'index'])

    
    if filter_years:
        df = df[[y in filter_years for y in df['year']]]
    if filter_yield:
        df = df[np.isnan(df['yield']) == False]
    if filter_wheat:
        df = df[df['grdkod_mar'] == 4]

    return df, correlation_filters




def get_norm_info(df, target, predictors):
    # normalization
    mX = np.mean(df[predictors].values, axis=0)
    sX = np.std(df[predictors].values, axis=0)
    sX[sX == 0] = 1

    mY = np.nanmean(df[target].values, axis=0)
    sY = np.nanstd(df[target].values, axis=0)
    
    return {
        'X': {
            'mean': mX,
            'std': sX
        },
        'y': {
            'mean': mY,
            'std': sY
        }
    }
def unnormalizeY(data, norm_info):
    return data*norm_info['y']['std'] + norm_info['y']['mean']
    


def connect_futurum(sub_df, edge_index, edge_features, dist_thresh=190):

    # Calculate distance between all points
    dx = (sub_df['x'].values.reshape((-1, 1)) - sub_df['x'].values.reshape((1, -1)))
    dy = (sub_df['y'].values.reshape((-1, 1)) - sub_df['y'].values.reshape((1, -1)))
    sqd = dx**2 + dy**2
    
    # Connect neighbouring points in space
    valid = np.where((sqd <= dist_thresh))
    for i in range(len(valid[0])):
        i1 = valid[0][i]
        i2 = valid[1][i]
        edge_index.append([sub_df.index[i1], sub_df.index[i2]])
        sqdist = max(sqd[i1, i2], 1)
        edge_features.append([10/np.sqrt(sqdist)])#, 0])
    
    return edge_index, edge_features

def create_futurum_dataset(df, target, predictors, target_year, pred_years, norm_info, device, dist_thresh=200, include_current_weeks=None, sparsity_thresh=None):
    dataset = []

    for pid in tqdm_notebook.tqdm_notebook(set(df['matching_pid'])):
        
        # Extract target year
        pdf = df[df['matching_pid'] == pid]
        pdf_target = pdf[pdf['year'] == target_year]
        
        # Check if yield exists 2019
        if np.mean(np.isnan(pdf_target['yield'])) == 1: continue

        # Extract previous years
        pdfs_pred = []
        for year in pred_years:
            pdfs_pred.append(pdf[pdf['year'] == year])

        # Find existing (x,y) coordinates and assign as index for quick retrieval
        pdf_target.index = [(pdf_target.loc[i, 'x'], pdf_target.loc[i, 'y']) for i in pdf_target.index]
        for pdf_pred in pdfs_pred:
            pdf_pred.index = [(pdf_pred.loc[i, 'x'], pdf_pred.loc[i, 'y']) for i in pdf_pred.index]

        # Find all (x,y) that exist in all years and that have yield in the target year
        valid = set(pdf_target[np.isnan(pdf_target['yield']) == False].index)
        for pdf_pred in pdfs_pred:
            valid = valid.intersection(set(pdf_pred.index))
        valid = list(valid)
        
        # If no such points exist, skip this field
        if len(valid) == 0: continue

        # Extract all valid points from each year, in the same order
        df_target = pdf_target.loc[valid]
        dfs_pred = []
        for pdf_pred in pdfs_pred:
            dfs_pred.append(pdf_pred.loc[valid])

        # Reset the indices to integers
        df_target.index = range(df_target.shape[0])
        for df_pred in dfs_pred:
            df_pred.index = range(df_pred.shape[0])

        # Create the feature vector, each point is an (x,y) point with all predicitor features stacked
        # Note that the vertices are not (x,y,t), but instead that all features for all years are stacked into a single vertex
        # This is because the output is just one yield per coordinate, so we cannot have one vertex per year and coordinate

    
        pred_features = [
            (df_pred[predictors].values - norm_info['X']['mean'])/norm_info['X']['std']
            for df_pred in dfs_pred
        ]
        if include_current_weeks is not None:
            week_indices = [i for i, p in enumerate(predictors) if (not p.startswith("week")) or (int(p.split("_")[1]) <= include_current_weeks)]
            week_predictors = [p for i, p in enumerate(predictors) if (not p.startswith("week")) or (int(p.split("_")[1]) <= include_current_weeks)]
            pred_features += [(df_target[week_predictors].values - norm_info['X']['mean'][week_indices])/norm_info['X']['std'][week_indices]]
        x = np.concatenate(pred_features, axis=1)
        # Create edges based on an arbitrary year. They all have the same (x,y) points so it doesnt matter.
        edge_index, edge_features = connect_futurum(dfs_pred[0], [], [], dist_thresh=dist_thresh)

        if (sparsity_thresh is not None) and len(edge_index)/pdf.shape[0] < sparsity_thresh: continue

        # Send tensor to device and save to dataset
        edge_index = torch.tensor(np.array(edge_index).transpose(), dtype=torch.long).to(device)
        edge_features = torch.tensor(np.array(edge_features), dtype=torch.float).to(device)
        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor((df_target[[target]].values - norm_info['y']['mean'])/norm_info['y']['std'], dtype=torch.float).to(device)

        X = torch_geometric.data.Data(
            x=x, 
            edge_index=edge_index, 
            edge_features=edge_features
        )
        
        info = {
            'pid': pid,
            'x': dfs_pred[0]['x'], 
            'y': dfs_pred[0]['y'], 
        }

        dataset.append((X, y, info))
    return dataset


def create_futurum_dataset_2(df, target, predictors, target_year, pred_years, norm_info, device, dist_thresh=200, include_current_weeks=None, sparsity_thresh=None):
    dataset = []

    full_data = 0
    lost_data = 0
    for pid in tqdm_notebook.tqdm_notebook(set(df['matching_pid'])):
        
        # Extract target year
        pdf = df[df['matching_pid'] == pid]
        pdf_target = pdf[pdf['year'] == target_year]
        
        # Check if yield exists 2019
        if np.mean(np.isnan(pdf_target['yield'])) == 1: continue

        # Extract previous years
        pdfs_pred = []
        for year in pred_years:
            pdfs_pred.append(pdf[pdf['year'] == year])

        # Find existing (x,y) coordinates and assign as index for quick retrieval
        pdf_target.index = [(pdf_target.loc[i, 'x'], pdf_target.loc[i, 'y']) for i in pdf_target.index]
        for pdf_pred in pdfs_pred:
            pdf_pred.index = [(pdf_pred.loc[i, 'x'], pdf_pred.loc[i, 'y']) for i in pdf_pred.index]

        # Find all (x,y) that exist in all years and that have yield in the target year
        valid = set(pdf_target[np.isnan(pdf_target['yield']) == False].index)
        for pdf_pred in pdfs_pred:
            valid = valid.intersection(set(pdf_pred.index))
        valid = list(valid)
        
        # If no such points exist, skip this field
        if len(valid) == 0: continue
        
        l1 = len(pdf_target[np.isnan(pdf_target['yield']) == False])
        l2 = len(valid)
        if l1 > 0 and l1 == l2: full_data += 1
        if l1 > 0 and l2 < l1: lost_data += l1-l2

        # Extract all valid points from each year, in the same order
        df_target = pdf_target.loc[valid]
        dfs_pred = []
        for pdf_pred in pdfs_pred:
            dfs_pred.append(pdf_pred.loc[valid])

        # Reset the indices to integers
        df_target.index = range(df_target.shape[0])
        for df_pred in dfs_pred:
            df_pred.index = range(df_pred.shape[0])

        # Create the feature vector, each point is an (x,y) point with all predicitor features stacked
        # Note that the vertices are not (x,y,t), but instead that all features for all years are stacked into a single vertex
        # This is because the output is just one yield per coordinate, so we cannot have one vertex per year and coordinate

    
        pred_features = [
            (df_pred[predictors].values - norm_info['X']['mean'])/norm_info['X']['std']
            for df_pred in dfs_pred
        ]
        if include_current_weeks is not None:
            week_indices = [i for i, p in enumerate(predictors) if (not p.startswith("week")) or (int(p.split("_")[1]) <= include_current_weeks)]
            week_predictors = [p for i, p in enumerate(predictors) if (not p.startswith("week")) or (int(p.split("_")[1]) <= include_current_weeks)]
            pred_features += [(df_target[week_predictors].values - norm_info['X']['mean'][week_indices])/norm_info['X']['std'][week_indices]]
        x = np.concatenate(pred_features, axis=1)
        # Create edges based on an arbitrary year. They all have the same (x,y) points so it doesnt matter.
        edge_index, edge_features = connect_futurum(dfs_pred[0], [], [], dist_thresh=dist_thresh)

        if (sparsity_thresh is not None) and len(edge_index)/pdf.shape[0] < sparsity_thresh: continue

        # Send tensor to device and save to dataset
        edge_index = torch.tensor(np.array(edge_index).transpose(), dtype=torch.long).to(device)
        edge_features = torch.tensor(np.array(edge_features), dtype=torch.float).to(device)
        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor((df_target[[target]].values - norm_info['y']['mean'])/norm_info['y']['std'], dtype=torch.float).to(device)

        X = torch_geometric.data.Data(
            x=x, 
            edge_index=edge_index, 
            edge_features=edge_features
        )
        
        info = {
            'pid': pid,
            'x': dfs_pred[0]['x'], 
            'y': dfs_pred[0]['y'], 
        }

        dataset.append((X, y, info))
        
    print(full_data, lost_data)
    return dataset

# time_scheme = None, "distance", "one-hot", "concat"
def connect_present(sub_df, edge_index, edge_features, dist_thresh=190, time_scheme=None):
    for year in set(sub_df['year']):
        year_df = sub_df[sub_df['year'] == year]

        dx = (year_df['x'].values.reshape((-1, 1)) - year_df['x'].values.reshape((1, -1)))
        dy = (year_df['y'].values.reshape((-1, 1)) - year_df['y'].values.reshape((1, -1)))
        dt = np.abs(year_df['year'].values.reshape((-1, 1)) - year_df['year'].values.reshape((1, -1)))
        sqd = dx**2 + dy**2
        
        # Connect in space same year
        valid = np.where((sqd <= dist_thresh) & (dt == 0))
        for i in range(len(valid[0])):
            i1 = valid[0][i]
            i2 = valid[1][i]
            edge_index.append([year_df.index[i1], year_df.index[i2]])
            sqdist = max(sqd[i1, i2], 1)
            features = [10/np.sqrt(sqdist)]

            if time_scheme == "one-hot":
                features += [0]*12
            elif time_scheme == "distance":
                features += [0]
            edge_features.append(features)
            
    # Connect in time, same point
    if time_scheme in ['one-hot', 'distance']:
        dx = (sub_df['x'].values.reshape((-1, 1)) == sub_df['x'].values.reshape((1, -1)))
        dy = (sub_df['y'].values.reshape((-1, 1)) == sub_df['y'].values.reshape((1, -1)))
        match = dx*dy
        
        valid = np.where((match == 1))
        for i in range(len(valid[0])):
            i1 = valid[0][i]
            i2 = valid[1][i]
            dt = sub_df.loc[i1, 'year'] - sub_df.loc[i2, 'year']
            if dt != 0:
                edge_index.append([sub_df.index[i1], sub_df.index[i2]])
                features = [0]
                if time_scheme == "one-hot":
                    time_features = [0]*12
                    time_features[1+dt+5] = 1
                    features += time_features
                elif time_scheme == "distance":
                    features += [1/dt]
                edge_features.append(features)
    
    return edge_index, edge_features

def create_present_dataset(df, target, predictors, norm_info, device, dist_thresh=190, time_scheme=None, size_thresh=None, sparsity_thresh=None):
    dataset = []
    skipped = []

    pbar = tqdm(set(df['matching_pid']))
    for pid in pbar:
        pdf = df[df['matching_pid'] == pid]
        #for year in set(pdf_base['year']):
        #    pdf = pdf_base[pdf_base['year'] == year]
        pdf = pdf.reset_index()
        if 'level_0' in pdf.columns:
            pdf = pdf.drop(columns=['level_0', 'index'])
        
        pbar.set_description_str(str(pid) + " (" + str(pdf.shape[0]) + ")")
        edge_index = []
        edge_features = []
        
        if size_thresh is None or pdf.shape[0] < size_thresh:
            edge_index, edge_features = connect_present(pdf, edge_index, edge_features, dist_thresh=dist_thresh, time_scheme=time_scheme)
        else:
            skipped.append((pid, pdf.shape[0]))
            continue
        
        
        if (sparsity_thresh is not None) and len(edge_index)/pdf.shape[0] < sparsity_thresh: 
            print("Failed", len(edge_index)/pdf.shape[0], sparsity_thresh)
            continue
        
        edge_index = torch.tensor(np.array(edge_index).transpose(), dtype=torch.long).to(device)
        edge_features = torch.tensor(np.array(edge_features), dtype=torch.float).to(device)
        x = torch.tensor((pdf[predictors].values - norm_info['X']['mean'])/norm_info['X']['std'], dtype=torch.float).to(device)
        y = torch.tensor((pdf[[target]].values - norm_info['y']['mean'])/norm_info['y']['std'], dtype=torch.float).to(device)
        valid = np.isnan(pdf[target]) == False
        

        X = torch_geometric.data.Data(
            x=x, 
            edge_index=edge_index, 
            edge_features=edge_features
        )
        
        info = {
            'pid': pid,
            'x': pdf['x'], 
            'y': pdf['y'], 
            'year': pdf['year'],
            'valid': valid
        }

        dataset.append((X, y, info))
    return dataset, skipped







def split_data(dataset, p_val, p_test, seed=None):
    
    if seed is not None: np.random.seed(seed)
    
    remainder = [d for d in dataset]

    N = np.sum([d[1].shape[0] for d in remainder])
    data_train = []
    data_test = []

    n_val = 0
    data_val = []
    while n_val < p_val*N:
        idx = np.random.randint(len(remainder))
        data = remainder[idx]
        data_val.append(data)
        remainder.remove(data)
        n_val += data[1].shape[0]


    n_test = 0
    data_test = []
    while n_test < p_test*N:
        idx = np.random.randint(len(remainder))
        data = remainder[idx]
        data_test.append(data)
        remainder.remove(data)
        n_test += data[1].shape[0]

    #print(N, n_val, n_test, str(np.round(n_val/N*100, 3)) + "%", str(np.round(n_test/N*100, 3)) + "%")

    data_train = remainder
    return data_train, data_val, data_test

def save_score_to_json(score, filename):
    save_str = "[\n"
    for run in score:
        save_str += "    [\n"
        for line in run:
            save_str += "        " + str(line)
            if line == run[-1]: save_str += "\n"
            else: save_str += ",\n"
            
        if run == score[-1]: save_str += "    ]\n"
        else: save_str += "    ],\n"
    save_str += "]"
    with open(filename, "w") as outfile:
        outfile.write(save_str)


def update_lr(optimizer, lr_schedule, rmse):
    def set_lr(lr):
        for g in optimizer.param_groups: 
            g['lr'] = lr

    for rmse_lim, lr in lr_schedule:
        if rmse < rmse_lim:
            set_lr(lr)
            return 
    set_lr(lr_schedule[-1][0])
def eval_error(pred, y, info, norm_info):
    yp = pred.cpu().detach().numpy().reshape(-1)
    yt = y.cpu().detach().numpy().reshape(-1)
    #if 'valid' in info: yt[info['valid'] == False] = np.nan
    err = (yt - yp)*norm_info['y']['std']
    return err

def eval_dataset(data, model, norm_info):
    err = np.zeros(0)
    for X, y, info in data:
        pred = model(X)
        err = np.concatenate([err, eval_error(pred, y, info, norm_info)])
    rmse = np.sqrt(np.nanmean(err**2))
    return rmse, err


def Loss_MSE(output, target):
    loss = torch.nanmean((output - target)**2)
    return loss

def Loss_MSE_no_mean(output, target):
    loss = torch.nanmean((output - (target - torch.nanmean(target) + torch.nanmean(output)))**2)
    return loss

def Loss_MSE_separate_mean(t):
    return lambda output, target: t*torch.nanmean((output - (target - torch.nanmean(target) + torch.nanmean(output)))**2) + (1-t)*(torch.nanmean(target) - torch.nanmean(output))**2

def run_model(p_val, p_test, dataset, norm_info, device, layer_sizes, seed=None, pred_var=False, lr_schedule=None, dropout=0.0, epochs=200, loss_function=Loss_MSE):
    
    if seed is not None: 
        print("Setting seed:", seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    if lr_schedule is not None:
        lr_schedule = sorted(lr_schedule)
    
    print(layer_sizes)
    model = GCN_v2(layer_sizes, dropout=dropout, edge_dim=dataset[0][0].edge_features.shape[1])
    model.to(device)

    history = {
        'train': [],
        'val': [],
        'test': []
    }
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = loss_function

    scores = {}
    top_score = 10000
    
    data_train, data_val, data_test = split_data(dataset, p_val, p_test, seed=seed)

    # Training loop
    model.train()

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        
        random.shuffle(data_train)
        err = np.zeros(0)
        total_los = []
        for X, y, info in data_train:
            optimizer.zero_grad()
            pred = model(X)
            y_cpu = y.clone().cpu().detach().numpy()
            pred_cpu = pred.clone().cpu().detach().numpy()
            if pred_var:
                y_cpu = y_cpu - np.mean(y_cpu) + np.mean(pred_cpu)
            y_cpu[np.isnan(y_cpu)] = pred_cpu[np.isnan(y_cpu)]
            #if 'valid' in info: y_cpu[info['valid'] == False] = pred_cpu[info['valid'] == False]
            y_gpu = torch.tensor(y_cpu).to(device)
            loss = criterion(pred, y_gpu)
            loss.backward()
            total_los.append(loss.item())
            optimizer.step()
            err = np.concatenate([err, eval_error(pred, y, info, norm_info)])
        train_rmse = np.sqrt(np.nanmean(err**2))
        history['train'].append(train_rmse)

        optimizer.zero_grad()

        val_rmse, val_err = eval_dataset(data_val, model, norm_info)
        #val_err = val_err - np.nanmean(val_err)
        history['val'].append(val_rmse)
        val_acc10 = np.mean((np.abs(val_err) < 1)[np.isnan(val_err) == False])
        metric = val_rmse
            
        test_rmse, test_err = eval_dataset(data_test, model, norm_info)
        #test_err = test_err - np.nanmean(test_err)
        #test_rmse = np.sqrt(np.nanmean(test_err**2))
        history['test'].append(test_rmse)
        acc5 = np.mean((np.abs(test_err) < 0.5)[np.isnan(test_err) == False])
        acc10 = np.mean((np.abs(test_err) < 1)[np.isnan(test_err) == False])
        acc20 = np.mean((np.abs(test_err) < 2)[np.isnan(test_err) == False])

        if lr_schedule is not None: 
            update_lr(optimizer, lr_schedule, val_rmse)

        if metric < top_score:
            top_score = metric
            torch.save(model, "/".join(["models", "current.pth"]))
            scores = {
                'test_rmse': test_rmse,
                'test_acc_0.5': acc5,
                'test_acc_1.0': acc10, 
                'test_acc_2.0': acc20,
                'test_rel_rmse': test_rmse/norm_info['y']['mean'],
            }

        pbar.set_description(f'Loss: {np.round(np.mean(total_los), 3)}, train rmse: {np.round(train_rmse, 3)}, val rmse: {np.round(val_rmse, 3)}, test rmse: {np.round(test_rmse, 3)}, top: [{np.round(metric, 3)} >= {np.round(top_score, 3)} {[np.round(scores[s], 3) for s in scores]}]')

    model = torch.load("models/current.pth")
    return model, history, scores, (data_train, data_val, data_test)

def run_model_fixed_sets(data_train, data_val, data_test, norm_info, device, layer_sizes, seed=None, pred_var=False, lr_schedule=None, dropout=0.0, epochs=200):
    
    if seed is not None: 
        print("Setting seed:", seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    if lr_schedule is not None:
        lr_schedule = sorted(lr_schedule)
    
    print(layer_sizes)
    model = GCN_v2(layer_sizes, dropout=dropout)
    model.to(device)

    history = {
        'train': [],
        'val': [],
        'test': []
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    scores = {}
    top_score = 10000

    # Training loop
    model.train()

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        
        random.shuffle(data_train)
        err = np.zeros(0)
        total_los = []
        for X, y, info in data_train:
            optimizer.zero_grad()
            pred = model(X)
            y_cpu = y.clone().cpu().detach().numpy()
            pred_cpu = pred.clone().cpu().detach().numpy()
            if pred_var:
                y_cpu = y_cpu - np.mean(y_cpu) + np.mean(pred_cpu)
            #y_cpu[np.isnan(y_cpu)] = pred_cpu[np.isnan(y_cpu)]
            #if 'valid' in info: y_cpu[info['valid'] == False] = pred_cpu[info['valid'] == False]
            y_gpu = torch.tensor(y_cpu).to(device)
            loss = criterion(pred, y_gpu)
            loss.backward()
            total_los.append(loss.item())
            optimizer.step()
            err = np.concatenate([err, eval_error(pred, y, info, norm_info)])
        train_rmse = np.sqrt(np.nanmean(err**2))
        history['train'].append(train_rmse)

        optimizer.zero_grad()

        val_rmse, val_err = eval_dataset(data_val, model, norm_info)
        #val_err = val_err - np.nanmean(val_err)
        history['val'].append(val_rmse)
        val_err = np.concatenate([(model(X).cpu().detach().numpy().reshape(-1) - y.cpu().detach().numpy().reshape(-1))[info['year'] == 2021] for X, y, info in data_val])*norm_info['y']['std']
        val_acc10 = np.mean((np.abs(val_err) < 1)[np.isnan(val_err) == False])
        metric = -val_acc10
            
        test_rmse, test_err = eval_dataset(data_test, model, norm_info)
        #test_err = test_err - np.nanmean(test_err)
        #test_rmse = np.sqrt(np.nanmean(test_err**2))
        history['test'].append(test_rmse)
        test_err = np.concatenate([(model(X).cpu().detach().numpy().reshape(-1) - y.cpu().detach().numpy().reshape(-1))[info['year'] == 2021] for X, y, info in data_test])*norm_info['y']['std']
        acc5 = np.mean((np.abs(test_err) < 0.5)[np.isnan(test_err) == False])
        acc10 = np.mean((np.abs(test_err) < 1)[np.isnan(test_err) == False])
        acc20 = np.mean((np.abs(test_err) < 2)[np.isnan(test_err) == False])

        if lr_schedule is not None: 
            update_lr(optimizer, lr_schedule, val_rmse)

        if metric < top_score:
            top_score = metric
            torch.save(model, "/".join(["models", "current.pth"]))
            scores = {
                'test_rmse': test_rmse,
                'test_acc_0.5': acc5,
                'test_acc_1.0': acc10, 
                'test_acc_2.0': acc20,
                'test_rel_rmse': test_rmse/norm_info['y']['mean'],
            }

        pbar.set_description(f'Loss: {np.round(np.mean(total_los), 3)}, train rmse: {np.round(train_rmse, 3)}, val rmse: {np.round(val_rmse, 3)}, test rmse: {np.round(test_rmse, 3)}, top: [{np.round(metric, 3)} >= {np.round(top_score, 3)} {[np.round(scores[s], 3) for s in scores]}]')

    model = torch.load("models/current.pth")
    return model, history, scores, (data_train, data_val, data_test)











class GCN_v1(torch.nn.Module):
    def __init__(self, sizes, dropout=0.0):
        super(GCN_v1, self).__init__()

        layers = []
        for i in range(len(sizes)-1):
            layers.append(
                torch_geometric.nn.conv.GATConv(sizes[i], sizes[i+1], dropout=dropout)  
            )

        self.convs = torch.nn.ModuleList(layers[:-1])
        self.output = layers[-1]
    
    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_features
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_features)
            x = torch.relu(x)
        x = self.output(x, edge_index, edge_features)
        return x

class GCN_v2(torch.nn.Module):
    def __init__(self, sizes, dropout=0.0, edge_dim=1):
        super(GCN_v2, self).__init__()

        layers = []
        for i in range(len(sizes)-1):
            layers.append(
                torch_geometric.nn.conv.GATv2Conv(sizes[i], sizes[i+1], dropout=dropout, edge_dim=edge_dim)  
            )

        self.convs = torch.nn.ModuleList(layers[:-1])
        self.output = layers[-1]

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_features
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_features)
            x = torch.relu(x)
        x = self.output(x, edge_index, edge_features)
        return x