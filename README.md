# Privacy Policies Classifier

To make the code work you will have to create a gold standard from the original dataset and place all the files into the raw_data folder. You can download the original data clicking [here](https://usableprivacy.org/static/data/OPP-115_v1_0.zip). The following paper explains how to transform the original data into a gold standard: https://usableprivacy.org/static/files/swilson_acl_2016.pdf. The website we are refering to is the following one: https://usableprivacy.org/data. 

You will also need to download the glove pretrained embeddings from [here](https://nlp.stanford.edu/projects/glove/). Download the one that says glove.6B.zip or click right [here](http://nlp.stanford.edu/data/glove.6B.zip). You will have to unzip the downloaded file and copy its contents into the folder glove.6B inside the cloned project. 

To run the code you will need to use CNN Example.ipynb as the main.py is still under development. Inside the notebook you will find detailed explanations on how to train the models and you will be able to evaluate the precision and recall after its training. 

Install:

pickle

