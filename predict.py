from cnn import CNN
from privacy_policies_dataset import PrivacyPoliciesDataset as PPD
from os.path import join, isfile
from os import listdir
from collections import OrderedDict
import re
import time
import torch
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt

def _cm(y, y_hat):
    
    #Empty dict where the data will be stored
    cm = OrderedDict()
    
    #Computation fo true positives, false positives, true negatives and false negatives
    tp = (y * y_hat).sum()
    tn = ((1 - y) * (1 - y_hat)).sum()
    fp = (y_hat * (1 - y)).sum()
    fn = ((1 - y_hat) * y).sum()
    
    #Storage of results in the dictionary
    cm['TP'] = tp.item()
    cm['TN'] = tn.item()
    cm['FP'] = fp.item()
    cm['FN'] = fn.item()
    
    return cm

def _cms(y, y_hat):
    
    #Empty tensor where the data will be stored
    cms = torch.tensor([])
    
    #Computation of cm for every label and pack them in cms
    for label in range(12):
        cm = torch.tensor(_cm(y[:,label], y_hat[:,label]).values()).unsqueeze(1)
        cms = torch.cat([cms,cm],1)
        
    return cms

def _metrics(cm):  
    
    tp, tn, fp, fn = cm.values()
    eps = 1e-10
    
    #Computation of F1 score, precision and recall
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    
    return f1, p, r

def _metrics_t(y, y_pred, t):
    
    y_hat = y_pred > t
    cm = _cm(y, y_hat.double())
    
    return _metrics(cm)    
    

def _metrics_wrt_t(y, y_pred):    

    #Initialization of range of thresholds and empty lists to store results
    ts = np.arange(0, 1, 0.01)
    f1s = []
    ps = []
    rs = []

    #loop that computes metrics for every threshold
    for t in ts:

        f1, p, r = _metrics_t(y, y_pred, t)
        
        #Storage of results
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        
    return f1s, ps, rs, ts

def _best_t_idx(y, y_pred):
    
    idxs = []

    for label in range(12):
       
        f1, p, r, ts = _metrics_wrt_t(y[:,label], y_pred[:,label])
        index = np.array(f1).argmax().item()
        idxs.append(index)

    return idxs

def best_ts(y, y_pred):
    
    ts = np.arange(0, 1, 0.01)
    idxs = _best_t_idx(y, y_pred)
    best_ts = [ts[i] for i in idxs]
    best_ts = torch.tensor(best_ts)
    return best_ts

def macro_metrics(y, y_hat):
    
    eps = 1e-10
    cms = _cms(y, y_hat)
    ps = cms[0] / (cms[0] + cms[2] + eps)
    rs = cms[0] / (cms[0] + cms[3] + eps)
    p = torch.mean(ps)
    r = torch.mean(rs)
    f1 = torch.mean(2 * ps * rs / (ps + rs + eps))
    
    return f1, p, r

def micro_metrics(y, y_hat):
    
    eps = 1e-10
    cms = _cms(y, y_hat)
    cm = cms.sum(1)
    p = cm[0] / (cm[0] + cm[2] + eps)
    r = torch.mean(cm[0] / (cm[0] + cm[3] + eps))
    f1 = torch.mean(2 * p * r / (p + r + eps))
    
    return f1, p, r

def best_metrics(y, y_pred):
    
    f1s = []
    ps = []
    rs = []
    ts = []
    idxs = _best_t_idx(y, y_pred)
    
    for idx, label in zip(idxs, range(12)):
        
        f1, p, r, t = _metrics_wrt_t(y[:,label], y_pred[:,label])
        f1s.append(f1[idx])
        ps.append(p[idx])
        rs.append(r[idx])
        ts.append(t[idx])
    
    return f1s, ps, rs, ts

def save_metrics(y, y_pred, path):
    
    def label_scores(y, y_pred, label, idx):          
    
        f1s, ps, rs, ts = _metrics_wrt_t(y[:,label], y_pred[:,label])
        best_scores = f1s[idx], ps[idx], rs[idx]
        scores_05 = _metrics_t(y[:,label], y_pred[:,label], 0.5)
        return scores_05 + best_scores
    
    round4 = lambda x: round(x,4)
    
    with open(path, 'w') as f:
        writer = csv.writer(f)
        idxs = _best_t_idx(y, y_pred)
        for label, idx in zip(range(12), idxs):
            scores = label_scores(y, y_pred, label, idx)
            scores = map(round4, scores)
            writer.writerows([scores])

def load_model(model_file, params_file):
    
    #We now load the parameters
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
        
    #We now load the model and pass the parameters
    model = CNN(**params)
    model.load_state_dict(torch.load(model_file))
    
    return model

def load_12CNN_model(path):
    
    def search_state(states, label):
    
        pattern = r'cnn_300_200_\[100, 25\]_1_\[3\]_\w*label{}_polisis_state.pt'
        pattern = pattern.format(label)
        for state in states:
            match = re.match(pattern, state)
            if match:
                return state
        
    def search_params(params, label):
    
        pattern = r'cnn_300_200_\[100, 25\]_1_\[3\]_\w*label{}_polisis_params.pkl'
        pattern = pattern.format(label)
        for param in params:
            match = re.match(pattern, param)
            if match:
                return param
    
    #We instantiate an empty dictionary that will contain the models
    model12cnn = OrderedDict()
    
    #Fetch the names of all the files
    states = [f for f in listdir(path) if isfile(join(path,f)) and '_state.pt' in f]
    params = [f for f in listdir(path) if isfile(join(path,f)) and '_params.pkl' in f]
    
    for label in range(12):
        model_file = search_state(states, label)
        model_file = join(path, model_file)
        params_file = search_params(params, label)
        params_file = join(path, params_file)
        model12cnn['model{}'.format(label)] = load_model(model_file, params_file)
        
    return model12cnn

def predict(data, models):
    
    #We instantiate an empty y and instantiate the x
    x = PPD.collate_data(data)[0]
    y = torch.tensor([])
    
    #We start a timer to compute predicions time and compute them
    start = time.time()
    for key, model in models.items():
        y_label = model(x)
        y = torch.cat([y, y_label],1)
    end = time.time()
        
    print("Prediction time: {} seconds". format(end - start))
    
    return y

def main():

    #We set the folder path containing the models and load the labels
    folder = 'trained_models/Multiclass'
    models = load_12CNN_model(folder)

    #We load the labels
    with open('labels.pkl') as f:
        labels = pickle.load(f)

    #We set the folder containing the data already prepared for predicting
    data_folder = 'datasets'
    data_file = join(data_folder, 'test_dataset_label6.pkl')

    #We load the data and get just the segments
    data = PPD.unpickle_dataset(data_file)

    #We predict the labels
    predictions = predict(data, models)
    
    #Computation of all metrics 
    f1s, ps, rs, ts = _metrics_wrt_t(data.labels_tensor,predictions)
    figure = plt.figure(figsize=(18,5))
    figure.suptitle('12-CNN Micro Averages with respect t')
    ax_f1 = figure.add_subplot(131)
    ax_f1.set_ylim(0,1)
    ax_p = figure.add_subplot(132)
    ax_p.set_ylim(0,1)
    ax_r = figure.add_subplot(133)
    ax_r.set_ylim(0,1)
    ax_f1.plot(ts, f1s)
    ax_p.plot(ts, ps)
    ax_r.plot(ts, rs)
    plt.show()

if __name__ == '__main__':

    main()
