import pandas as pd
import tqdm
import numpy as np
import json
import cv2
from datetime import datetime
from datetime import timedelta
import os
import xgboost as xgb
import matplotlib
import sklearn

import tensorflow as tf
from tensorflow import keras
from keras import layers
import torch
import torch_geometric
import random

def do_full_test(path, use_neural=False, save=False):
    df, correlation_filters = preprocess_data(path, save=save)
    results = evaluate_data(df, correlation_filters, use_neural=use_neural)
    display_results(results, df, correlation_filters)

def preprocess_data(path, save=False):

    dummy_year = 1990
    sentinel2_smoothing_channels = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    sentinel2_smoothing_kernels = [6,1,1,1,2,2,2,1,2,6,6,2,2]
    time_interpolation_start = "15/4"
    time_interpolation_end = "15/7"
    time_interpolation_step_days = 7
    time_zero = datetime(dummy_year, 3, 1)
    interpol_start = (datetime(dummy_year, int(time_interpolation_start.split("/")[1]), int(time_interpolation_start.split("/")[0])) - time_zero).days
    interpol_end = (datetime(dummy_year, int(time_interpolation_end.split("/")[1]), int(time_interpolation_end.split("/")[0])) - time_zero).days
    interpol_step = time_interpolation_step_days
    yield_smoothing_sigmas = [1,3,6]
    sen1_indices = [
        ["sarvi", lambda vh, vv: 4 * vh /( vv + vh)]
    ]
    sen2_indices = [
        ["ndvi", "B08", "B04", lambda c1, c2: (c1 - c2)/(c1 + c2)], 
        ["ndwi", "B03", "B08", lambda c1, c2: (c1 - c2)/(c1 + c2)], 
        ["ndbi", "B11", "B08", lambda c1, c2: (c1 - c2)/(c1 + c2)]
    ]
    category_features = ["topology"]
    correlation_filter_thresholds = [0.5, 0.9, 0.99]
    min_correlation = 0.1

    bar_format = '{bar}| {n_fmt}/{total_fmt} {desc} [elapsed: {elapsed}]'

    pbar = tqdm.notebook.tqdm(range(11), bar_format=bar_format)

    pbar.set_description_str("Reading File")
    df = pd.read_feather(path)
    pbar.update()

    pbar.set_description_str("Interpolating Sentinel 2 in Space")
    df = interpolate_channels(df, sentinel2_smoothing_channels, sentinel2_smoothing_kernels)
    pbar.update()


    # Linear Interpolation in time
    pbar.set_description_str("Interpolating Sentinel 1 in Time")
    sen1_df = interpolate_time(df, "sigma0", interpol_start, interpol_end, interpol_step, time_zero)
    pbar.update()

    pbar.set_description_str("Interpolating Sentinel 2 in Time")
    sen2_df = interpolate_time(df, "_B", interpol_start, interpol_end, interpol_step, time_zero)
    pbar.update()

    pbar.set_description_str("Interpolating Theta in Time")
    theta_chan = [c for c in df.columns if '_theta' in c]
    if len(theta_chan) > 0: 
        theta_df = interpolate_time(df, "_theta", interpol_start, interpol_end, interpol_step, time_zero)
    pbar.update()

    df = pd.concat([df, sen1_df, sen2_df, theta_df], axis=1)
    df = df[[c for c in df.columns if ('_B' not in c or '_i' in c) and ('sigma' not in c or '_i' in c) and ('_theta' not in c or '_i' in c)]]
    df.columns = [c[:-2] if c.endswith("_i") else c for c in df.columns]
    
    
    
    # Rename satelite data columns to 'week' format instead of 'date'
    replaces = []
    weeks = set(["_".join(d.split("_")[:2]) + "_" for d in df.columns if "week_" in d])
    for week in weeks:
        replaces += [(d, d.replace(week, week + "weather_")) for d in df.columns if d.startswith(week)]
    dates = [d.split("_")[0] for d in df.columns if "sigma0_vh_norm_multi" in d]
    weeks = ["week_" + str(week) for week in range(16, 16+len(dates))]
    for i in range(len(weeks)):
        replaces += [(d, d.replace(dates[i], weeks[i])) for d in df.columns if d.startswith(dates[i])]
    replaces = {r[0]: r[1] for r in replaces}
    df = df.rename(replaces, axis=1)
    
    
    # Try to fill in features with Nan-values
    pbar.set_description_str("Removing Nan-values")
    df = fill_empty_fields(df, 5)
    pbar.update()

    # Smooth Yield
    pbar.set_description_str("Smooth Yield")
    df = yield_smoothing(df, yield_smoothing_sigmas)
    pbar.update()

    # Calculate indices
    pbar.set_description_str("Calculate Sentinel 2 Indices")
    df = create_sentinel_indices(df, sen1_indices, sen2_indices)
    pbar.update()
    
    # One-hot encode the Topolgoy feature
    pbar.set_description_str("One-hot Encode Category Labels")
    df = pd.get_dummies(df, columns=category_features, dtype=int)
    pbar.update()

    # Do feature importance analysis
    pbar.set_description_str("Filter Features by Yield Correlation")
    correlation_filters = filter_features(df[np.isnan(df['yield']) == False], correlation_filter_thresholds, min_correlation)
    pbar.update()

    pbar.set_description_str("Saving")
    # save
    if save:
        os.makedirs("data/full_data/processed", exist_ok=True)
        folder_name = "data/full_data/processed"
        if type(save) == str:
            file_name = save
        else: 
            file_name = path.split("/")[-1].split(".")[0]
        df.to_feather(folder_name + "/" + file_name + ".feather")
        f = open(folder_name + "/" + file_name + ".json", "w")
        json.dump(correlation_filters, f)
        f.close()
    pbar.update()

    return df, correlation_filters

def evaluate_data(df, correlation_filters, use_neural=False):

    print("Evaluating data")
    results = {}
    filt = split_data(df, 0.1)
    targets = [d for d in df.columns if "yield" in d]
    models = [True, False] if use_neural else [False]

    for model in models:
        for corr_type in correlation_filters:
            corr_filter = correlation_filters[corr_type]
            predictors = [c for c in corr_filter if c not in ['index', 'x', 'y', 'x_int', 'y_int', 'polygon_id'] and "yield" not in c]

            for target in targets:
                run_id = target + "&" + corr_type + "&" + ('neural' if model else 'xgb')

                print(run_id)

                for i in range(1):

                    polygon_info = {}
                    y_norm = np.zeros(df.shape[0])
                    for pid in set(df['polygon_id']):
                        y = df[df['polygon_id'] == pid][target]
                        m = np.mean(y)
                        s = np.std(y)
                        polygon_info[pid] = [m, s]
                        y = ((y-m)).values
                        y_norm[df['polygon_id'] == pid] = y

                
                    res_df = pd.DataFrame()
                    res_df['x'] = df['x'][filt >  0]
                    res_df['y'] = df['y'][filt >  0]
                    res_df['polygon_id'] = df['polygon_id'][filt >  0]

                    # Predict Full Yield
                    Xt = df[predictors][filt == 0]
                    Xv = df[predictors][filt >  0]
                    yt = df[target][filt == 0]
                    yv = df[target][filt >  0]
                    true, pred = train_and_predict(Xt, yt, Xv, yv, model)

                    res_df['full_true'] = true
                    res_df['full_pred'] = pred
                    for pid in set(df['polygon_id']):
                        res_df.loc[res_df['polygon_id'] == pid, 'full_true'] -= polygon_info[pid][0]
                        res_df.loc[res_df['polygon_id'] == pid, 'full_pred'] -= polygon_info[pid][0]
                        #res_df.loc[res_df['polygon_id'] == pid, 'full_true'] /= polygon_info[pid][1]
                        #res_df.loc[res_df['polygon_id'] == pid, 'full_pred'] /= polygon_info[pid][1]
                    pred = res_df['full_true']
                    true = res_df['full_pred']

                    # Predict Deviation
                    Xt = df[predictors][filt == 0]
                    Xv = df[predictors][filt >  0]
                    yt = y_norm[filt == 0]
                    yv = y_norm[filt >  0]
                    true, pred = train_and_predict(Xt, yt, Xv, yv, model)
                    res_df['dev_true'] = true
                    res_df['dev_pred'] = pred

                    results[run_id] = res_df
    return results

def display_results(results, df, correlation_filters):
    targets = [d for d in df.columns if 'yield' in d]
    corrs = list(correlation_filters.keys())
    M = len(targets)
    N = len(corrs)
    metrics = {
        'rmse': {
            "reversed": True, 
            "range": [0,2],
            "xgboost_dev": np.zeros((M,N)),
            "xgboost_full": np.zeros((M,N)) ,
            "neural_dev": np.zeros((M,N)),
            "neural_full": np.zeros((M,N)) 
        },
        'r2': {
            "reversed": False, 
            "range": [-1,1],
            "xgboost_dev": np.zeros((M,N)),
            "xgboost_full": np.zeros((M,N)) ,
            "neural_dev": np.zeros((M,N)),
            "neural_full": np.zeros((M,N)) 
        },
        'acc': {
            "reversed": False, 
            "range": [.5,1],
            "xgboost_dev": np.zeros((M,N)),
            "xgboost_full": np.zeros((M,N)),
            "neural_dev": np.zeros((M,N)),
            "neural_full": np.zeros((M,N)) 
        },
    }
    for settings in list(results.keys()):
        r = targets.index(settings.split("&")[0])
        c = corrs.index(settings.split("&")[1])
        neural = settings.split("&")[2]
        prefix = "neural_" if neural == "neural" else "xgboost_"
        res_df = results[settings]
        true = res_df['dev_true']

        pred = res_df['full_pred']
        abserr = np.abs(true - pred)
        metrics['rmse'][prefix + 'full'][r, c] = np.sqrt(np.mean(abserr**2))
        metrics['r2'][prefix + 'full'][r, c] = sklearn.metrics.r2_score(true, pred)
        metrics['acc'][prefix + 'full'][r, c] = np.mean(abserr < 1)

        pred = res_df['dev_pred']
        abserr = np.abs(true - pred)
        metrics['rmse'][prefix + 'dev'][r, c] = np.sqrt(np.mean(abserr**2))
        metrics['r2'][prefix + 'dev'][r, c] = sklearn.metrics.r2_score(true, pred)
        metrics['acc'][prefix + 'dev'][r, c] = np.mean(abserr < 1)

    fields = ["xgboost_full", "xgboost_dev", "neural_full", "neural_dev"]
    metrics_list = []
    for metric in metrics:
        for field in fields:
            if (metrics[metric][field][0,0] == 0): continue
            df_metric = pd.DataFrame(metrics[metric][field])
            df_metric.columns = corrs
            df_metric.index = targets
            df_metric.columns.name = field + " " + metric
            df_metric.columns.name += " "*(20 - len(df_metric.columns.name))
            cmap = matplotlib.colormaps["RdYlGn"]
            if metrics[metric]['reversed']: cmap = cmap.reversed()
            df_style = df_metric.style.background_gradient(cmap=cmap, vmin=metrics[metric]['range'][0], vmax=metrics[metric]['range'][1])
            metrics_list.append(df_style)
    

    path = open('results.csv', 'w')
    for metric in metrics_list:
        path.write(str(metric.data))
        path.write("\n")
        path.write("\n")
    path.close()
    #path.write_text(s.to_string(delimiter=','))
    return metrics_list



def split_data(df, p, seed=-1):
    if seed > 0: np.random.seed(seed)

    polygons = list(set(df['polygon_id']))

    filt = df['polygon_id'] < 0
    picked = []
    while np.mean(filt>0) < p:
        idx = np.random.randint(len(polygons))
        count = 0
        while idx in picked and count < 1000:
            idx = np.random.randint(len(polygons))
            count += 1
        if count == 1000:
            print("ERROR WHEN PICKING FIELDS")
        picked.append(idx)
        print("added field", idx)
        filt += df['polygon_id'] == polygons[idx]
    return filt

def train_and_predict(Xt, yt, Xv, yv, neural):
    
    mX = np.mean(Xt, axis=0)
    sX = np.std(Xt, axis=0)
    sX[sX == 0] = 1
    Xt = (Xt-mX)/sX
    Xv = (Xv-mX)/sX

    my = np.mean(yt)
    sy = np.std(yt)
    yt = (yt-my)/sy
    yv = (yv-my)/sy

    if not neural:
        model = xgb.XGBRegressor(random_state=0)
        model.fit(Xt, yt)
        yp = model.predict(Xv)
    else: 
        model = keras.Sequential([layers.Dense(sz, activation='relu', kernel_regularizer='l2') for sz in [64,64,32,16]] + [
            layers.Dense(1, kernel_regularizer='l2')
        ])
        model.compile(loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(0.001),
        )
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
        )]
        history = model.fit(
            Xt,
            yt,
            validation_data=(Xv, yv), 
            epochs=100,
            callbacks=callbacks
        )

        # Predict
        yp = model.predict(Xv)
    
    yv = yv*sy + my
    yp = yp*sy + my
    yp = yp.reshape(-1)
    return yv, yp



def interpolate_channels(df, channels, kernel_sizes):
    with tqdm.notebook.tqdm(total=len(channels), leave=False, bar_format='{bar}| {n_fmt}/{total_fmt} {desc} [{elapsed}<{remaining}]') as pbar:
        for ci, channel in enumerate(channels):
            pbar.update()
            pbar.set_description(channel)
            ksize = kernel_sizes[ci]
            if ksize == 1: continue

            cols = [c for c in df.columns if channel in c]
            
            plot = -2
            for pid in list(set(df['polygon_id'])):

                polygon = df[df['polygon_id'] == pid].copy()

                polygon['x_int'] = np.floor(polygon['x']/10)
                polygon['x_int'] -= np.min(polygon['x_int'])
                polygon['x_int'] = polygon['x_int'].values.astype(int)
                polygon['y_int'] = np.floor(polygon['y']/10)
                polygon['y_int'] -= np.min(polygon['y_int'])
                polygon['y_int'] = polygon['y_int'].values.astype(int)

                img = np.zeros((int(max(polygon['x_int']))+1, int(max(polygon['y_int']))+1, len(cols)))

                img[list(polygon['x_int'].astype(int)), list(polygon['y_int'].astype(int)), :] = polygon[cols].values

                chan = 0
                # smoothing
                blur = np.ones((ksize,ksize))/(ksize**2)
                blurred = np.zeros(img.shape)
                for chan in range(len(cols)):   
                    blurred[:, :, chan] = cv2.filter2D(img[:, :, chan], -1, blur)

                kernel_count = np.zeros(img.shape)
                for chan in range(len(cols)):   
                    kernel_count[:, :, chan] = cv2.filter2D(1.0*(img[:, :, chan] != 0), -1, blur)
                kernel_count[kernel_count == 0] = 1

                smoothed = (blurred/kernel_count)
                smoothed[img == 0] = np.nan

                polygon[cols] = smoothed[list(polygon['x_int'].astype(int)), list(polygon['y_int'].astype(int)), :]
                df.loc[df['polygon_id'] == pid, cols] = polygon[cols]
                
                plot += 1
    pbar.close()
    return df

def interpolate_time(df, identifier, start, end, step, time_zero):
    channels = sorted(list(set(["_".join(c.split("_")[1:]) for c in df.columns if identifier in c])))
    sen_df = pd.DataFrame()
    index = 1
    with tqdm.notebook.tqdm(total=len(channels), leave=False, bar_format='{bar}| {n_fmt}/{total_fmt} {desc} [{elapsed}<{remaining}]') as pbar:
        for channel in channels:

            # extract channel dates
            cols = sorted(list([c for c in df.columns if channel in c]))
            mini_df = df[cols]

            # linearly interpolate nan values
            mini_df.columns = [(datetime.strptime(c.split("_")[0], "%m-%d") - datetime(1900, 3, 1)).days for c in cols]
            mini_df = mini_df.interpolate('index', limit_direction="both", axis=1)

            # shuffle in interpolation dates
            for days in range(start, end, step):
                mini_df[days + 0.01] = np.nan
            mini_df = mini_df[sorted(mini_df.columns)]

            # interpolate new values
            mini_df = mini_df.interpolate('index', limit_direction="both", axis=1)

            # convert to string columns
            mini_df = mini_df[[c for c in mini_df.columns if c%1 > 0.005]]
            mini_df.columns = [(time_zero + timedelta(days=round(c))).strftime("%m-%d") + "_" + channel + "_i" for c in mini_df.columns]

            sen_df = pd.concat([sen_df, mini_df], axis=1)
            index += 1
            pbar.update()
            pbar.set_description(channel)
    pbar.close()
    return sen_df

def clear_nan(df):
    nan_fields = {c: np.sum(np.isnan(df[c])) for c in df.columns if (np.sum(np.isnan(df[c])) > 0) and (c != "yield")}
    print("There are nan values in these columns", nan_fields)
    print("Removing", np.sum(np.sum(np.isnan(df[[c for c in df.columns if "yield" not in c]]), axis=1) > 0), "data points")
    df = df[np.sum(np.isnan(df[[c for c in df.columns if "yield" not in c]]), axis=1) == 0]
    return df

def fill_empty_fields(df, K):
    nan_fields = {c: np.sum(np.isnan(df[c])) for c in df.columns if (np.sum(np.isnan(df[c])) > 0) and ("yield" not in c)}
    cols = list(nan_fields.keys())

    for pid in tqdm.notebook.tqdm(set(df['polygon_id']), leave=False):
        pdf = df[df['polygon_id'] == pid]
        val_df = pdf[np.isnan(pdf[cols[0]]) == False]

        for i in pdf[np.isnan(pdf[cols[0]])].index:
            x = pdf.loc[i, 'x']
            y = pdf.loc[i, 'y']
            
            d = np.sqrt((val_df.x - x)**2 + (val_df.y - y)**2).sort_values()
            d = d[1:min(K+1, len(d))]
            weight = 1/d
            
            for col in cols:
                val = np.sum(val_df.loc[d.index, col]*weight, axis=0)/np.sum(weight)
                df.loc[i, col] = val
    return df

def yield_smoothing(full_df, sigmas):
    for sigma in sigmas:
        full_df.loc[:, 'yield_' + str(sigma)] = np.nan*np.zeros(full_df.shape[0])
    
    for pid in set(full_df['polygon_id']):
        
        df = full_df[full_df['polygon_id'] == pid].copy()
        if np.sum(np.isnan(df['yield']) == False) == 0: continue
        
        x_int = np.floor(df.loc[:, 'x']/10)
        x_int -= np.min(x_int)
        x_int = x_int.astype(int)
        y_int = np.floor(df.loc[:, 'y']/10)
        y_int -= np.min(y_int)
        y_int = y_int.astype(int)

        full_y = np.zeros((int(max(x_int))+1, int(max(y_int))+1))
        valid_y = np.zeros((int(max(x_int))+1, int(max(y_int))+1))

        for i in df.index:
            x = x_int[i]
            y = y_int[i]
            valid_y[x, y] = 1
            
            val = df['yield'][i] 
            if np.isnan(val): val = 0
            full_y[x, y] = val
            
        for sigma in sigmas:
            ksize = sigma*4+1
            blur = np.zeros((ksize,ksize))
            blur[int(ksize/2),int(ksize/2)] = 1
            if sigma > 0:
                blur = cv2.GaussianBlur(blur, (ksize, ksize), sigma, borderType=0)

            blurred_y = cv2.filter2D(full_y, -1, blur)
            kernel_count = cv2.filter2D(1.0*(full_y > 0), -1, blur)
            kernel_count[kernel_count == 0] = 1
            smoothed_y = valid_y*(blurred_y/kernel_count)

            blurred_yield = np.zeros(len(df.index))
            for i in range(df.shape[0]):
                x = x_int.iloc[i]
                y = y_int.iloc[i]
                blurred_yield[i] = smoothed_y[x, y]
            full_df.loc[df.index, 'yield_' + str(sigma)] = blurred_yield
    return full_df

def create_sentinel_indices(df, sen1_indices, sen2_indices):
    def make_index(df, date, index, channel1, channel2, func):
        c1 =  df[date.replace("CHANNEL", channel1)]
        c2 =  df[date.replace("CHANNEL", channel2)]
        df[date.replace("CHANNEL", index)] = func(c1, c2)

    dates = [c.replace("vh", "CHANNEL").replace("vv", "CHANNEL") for c in df.columns if "sigma" in c]
    for date in dates:
        for index in sen1_indices:
            make_index(df, date, index[0], "vh", "vv", index[1])

    dates = [c.replace("B01", "CHANNEL") for c in df.columns if "B01" in c]
    for date in dates:
        for index in sen2_indices:
            make_index(df, date, index[0], index[1], index[2], index[3])
    return df

def filter_features(df, filter_thresholds, min_corr):
    
    df = df[np.isnan(df['yield']) == False]

    def filter_corr(corr_threshold, min_corr):
        cols = [d for d in df.columns if d not in ["index", "polygon_id", "x", "y", "x_int", "y_int", "yield", "year"] and np.std(df[d]) > 0 and np.abs(np.corrcoef(df['yield'], df[d])[0,1]) > min_corr]
        yield_corr = pd.DataFrame([np.abs(np.corrcoef(df['yield'], df[c])[0,1]) for c in cols])
        yield_corr = yield_corr.transpose()
        yield_corr.columns = cols
        yield_corr = yield_corr.sort_values(0, axis=1)
        cols = list(yield_corr.columns)

        #print("\n" + str(corr_threshold) + "\n")
        best_features = ["yield", "x", "y", "polygon_id"]
        
        N = len(cols)
        with tqdm.notebook.tqdm(total=N, desc=str(corr_threshold), leave=False, bar_format='{bar}| {n_fmt}/{total_fmt} {desc} [{elapsed}<{remaining}]') as pbar:
            while len(cols) > 1:
                col = cols[-1]
                score = yield_corr[col].values[0]
                #print(col, score)
                best_features.append(col)
                removals = []
                for c in cols:
                    if np.abs(np.corrcoef(df[col], df[c])[0,1]) > corr_threshold:
                        removals.append(c)

                for c in removals:
                    cols.remove(c)

                pbar.update(N - len(cols) - pbar.n)
        pbar.close()
        return best_features
    correlation_filters = {
        'base': list(df.columns),
    }
    for thresh in filter_thresholds:
        best_features = filter_corr(thresh, min_corr)
        correlation_filters["corr_" + str(thresh)] = best_features
    return correlation_filters




def split_graph_data(dataset, p_val, p_test):

    remainder = [d for d in dataset]
    N = np.sum([d[1].shape[0] for d in remainder])

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

    data_train = remainder
    return data_train, data_val, data_test

class GCN(torch.nn.Module):
    def __init__(self, predictors):
        super(GCN, self).__init__()
        self.conv1 = torch_geometric.nn.conv.GATConv(len(predictors), len(predictors)//2)
        self.conv2 = torch_geometric.nn.conv.GATConv(len(predictors)//2, 64)
        self.conv3 = torch_geometric.nn.conv.GATConv(64, 1)

    def forward(self, data):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_features
        x = self.conv1(x, edge_index, edge_features)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_features)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_features)
        return x

def GAT_run(year, target, corr_filter, p_val, p_test, K):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path = "data/full_data/processed/processed_" + str(year) + ".feather"
    #path = "data/processed/2019_export_date_2024-03-25.feather"
    df = pd.read_feather(path)
    f = open(path.replace(".feather", ".json"), "r")
    correlation_filters = json.load(f)
    f.close()

    if 'level_0' in df.columns:
        df = df.drop(columns=['level_0'])

    df = df[np.isnan(df['yield']) == False]

    pmap = np.array(list(set(df['polygon_id'])))
    polygons = [np.where(np.array(pmap) == i)[0][0] for i in df['polygon_id']]
    df['polygon_id'] = polygons

    df = df.reset_index()
    if 'level_0' in df.columns:
        df = df.drop(columns=['level_0', 'index'])




    predictors = [c for c in correlation_filters[corr_filter] if c not in ['index', 'x', 'y', 'x_int', 'y_int', 'polygon_id'] and "yield" not in c]

    # normalization
    mX = np.mean(df[predictors].values, axis=0)
    sX = np.std(df[predictors].values, axis=0)
    sX[sX == 0] = 1

    mY = np.mean(df[target].values, axis=0)
    sY = np.std(df[target].values, axis=0)

    dataset = []

    for pid in tqdm.tqdm(set(df['polygon_id'])):
        pdf = df[df['polygon_id'] == pid]
        pdf = pdf.reset_index()
        if 'level_0' in pdf.columns:
            pdf = pdf.drop(columns=['level_0', 'index'])

        edge_index = []
        edge_features = []
        dx = pdf['x'].values.reshape((-1, 1)) - pdf['x'].values.reshape((1, -1))
        dy = pdf['y'].values.reshape((-1, 1)) - pdf['y'].values.reshape((1, -1))
        sqd = dx**2 + dy**2
        valid = np.where((sqd > 0) & (sqd < 400))
        for i, j in zip(valid[0], valid[1]):
            edge_index.append([pdf.index[i], pdf.index[j]])
            edge_features.append([10/np.sqrt(sqd[i, j])])

        edge_index = torch.tensor(np.array(edge_index).transpose(), dtype=torch.long).to(device)
        edge_features = torch.tensor(np.array(edge_features), dtype=torch.float).to(device)
        x = torch.tensor((pdf[predictors].values - mX)/sX, dtype=torch.float).to(device)
        y = torch.tensor((pdf[[target]].values - mY)/sY, dtype=torch.float).to(device)
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
            'valid': valid
        }

        dataset.append((X, y, info))

    
    def eval_error(pred, y, info):
        yp = pred.cpu().detach().numpy().reshape(-1)
        yt = y.cpu().detach().numpy().reshape(-1)
        yt[info['valid'] == False] = np.nan
        err = (yt - yp)*sY
        return err

    def eval_dataset(data, model):
        err = np.zeros(0)
        for X, y, info in data:
            pred = model(X)
            err = np.concatenate([err, eval_error(pred, y, info)])
        rmse = np.sqrt(np.nanmean(err**2))
        return rmse, err
    
    def run_model(p_val, p_test):

        # Define Datasets
        data_train, data_val, data_test = split_graph_data(dataset, p_val, p_test)

        # Create Model
        model = GCN(predictors)
        model.to(device)

        # Setup Optimizer and Schedule
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=33, gamma=0.1)
        def update_lr(rmse):
            if rmse < 0.5:
                for g in optimizer.param_groups: 
                    g['lr'] = 0.00005
            elif rmse < 0.75:
                for g in optimizer.param_groups: 
                    g['lr'] = 0.0001
            elif rmse < 1:
                for g in optimizer.param_groups: 
                    g['lr'] = 0.0005
            elif rmse < 1.5:
                for g in optimizer.param_groups: 
                    g['lr'] = 0.001
            else:
                for g in optimizer.param_groups: 
                    g['lr'] = 0.001


        history = {
            'train': [],
            'val': []
        }
        top_score = 10000

        # Train model
        model.train()
        pbar = tqdm.tqdm(range(200), leave=False)
        for epoch in pbar:
            
            random.shuffle(data_train)
            err = np.zeros(0)
            total_los = []
            for X, y, info in data_train:
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                total_los.append(loss.item())
                optimizer.step()
                err = np.concatenate([err, eval_error(pred, y, info)])
            train_rmse = np.sqrt(np.nanmean(err**2))
            history['train'].append(train_rmse)

            optimizer.zero_grad()

            val_rmse, val_err = eval_dataset(data_val, model)
            history['val'].append(val_rmse)
            val_acc10 = np.mean((val_err < 1)[np.isnan(val_err) == False])
            metric = -val_acc10

            update_lr(val_rmse)

            if metric < top_score:
                top_score = metric
                torch.save(model, "/".join(["models", "current.pth"]))

            pbar.set_description(f'Loss: {np.round(np.mean(total_los), 3)}, train rmse: {np.round(train_rmse, 3)}, val rmse: {np.round(val_rmse, 3)}, top: [{np.round(top_score, 3)}]')

        model = torch.load("models/current.pth")

        
        # Evaluate on test set
        test_rmse, test_err = eval_dataset(data_test, model)
        acc5 = np.mean((test_err < 0.5)[np.isnan(test_err) == False])
        acc10 = np.mean((test_err < 1)[np.isnan(test_err) == False])
        acc20 = np.mean((test_err < 2)[np.isnan(test_err) == False])
        scores = {
            'test_rmse': test_rmse,
            'test_acc_0.5': acc5,
            'test_acc_1.0': acc10, 
            'test_acc_2.0': acc20,
        }
        print(scores)

        return model, history, scores

    #model, history, scores = run_model(p_val=0.15, p_test=0.15)

    results = []
    rmse = []
    acc = []
    for i in range(K):
        model, history, scores = run_model(p_val=p_val, p_test=p_test)
        rmse.append(scores['test_rmse'])
        acc.append(scores['test_acc_1.0'])
        results.append((model, history, scores))

    print("rmse:", np.mean(rmse) - 1.92*np.std(rmse)/np.sqrt(K), np.mean(rmse), np.mean(rmse) + 1.92*np.std(rmse)/np.sqrt(K))
    print("acc:", np.mean(acc) - 1.92*np.std(acc)/np.sqrt(K), np.mean(acc), np.mean(acc) + 1.92*np.std(acc)/np.sqrt(K))
    return results