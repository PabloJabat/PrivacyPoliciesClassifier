from cnn import CNN
from os.path import join, isfile
from os import listdir
import torch
import pickle
from privacy_policies_dataset import PrivacyPoliciesDataset as PPD

def load_model(path, label):

    #We make the imports needed for this function
    import pickle
    import torch
    from os.path import join

    #We set the name of the model and its parameters
    models_files = join(path, 'cnn_300_200_[100, 25]_1_[3]_label{}_polisis_state.pt')
    model_file = models_files.format(label)
    params_files = join(path, 'cnn_300_200_[100, 25]_1_[3]_label{}_polisis_params.pkl')
    params_file = params_files.format(label)

    #We now load the parameters
    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    #We now load the model and pass the parameters
    model = CNN(**params)
    model.load_state_dict(torch.load(model_file))

    return model

def load_12CNN_model(path):

    from collections import OrderedDict
    #We instantiate an empty dictionary that will contain the models
    model12cnn = OrderedDict()
    for label in range(12):
        model12cnn['model{}'.format(label)] = load_model(path, label)

    return model12cnn

def predict(data, models, threshold=0.5):

    #We import time to compute time of prediction
    import time
    import torch

    x = PPD.collate_data(data)[0]
    y = torch.tensor([])

    start = time.time()
    for key, model in models.items():
        y_label = model(x)
        y = torch.cat([y, y_label])
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

if __name__ == '__main__':

    main()

