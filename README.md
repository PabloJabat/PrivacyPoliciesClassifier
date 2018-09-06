# Towards-Mimicking-UsablePrivacy

To make the code work you will have to create a gold standard from the original dataset and place all the files into the raw_data folder. You can download the original data clicking [here](https://usableprivacy.org/static/data/OPP-115_v1_0.zip). The following paper explains how to transform the original data into a gold standard: https://usableprivacy.org/static/files/swilson_acl_2016.pdf. The website we are refering to is the following one: https://usableprivacy.org/data. 

You will also need to download the glove pretrained embeddings from [here](https://nlp.stanford.edu/projects/glove/). Download the one that says glove.6B.zip or click right [here](http://nlp.stanford.edu/data/glove.6B.zip). You will have to unzip the downloaded file and copy its contents into the folder glove.6B inside the cloned project. 

To run the example code just run:

`python main.py`

This code will show you an example of serializing a sentence into integers and output a set of all the words included in the gold standard. It will also create a file called dictionary.pkl with all the information of that set. The main will read directly from that file as long as it exist. This will make the code run much faster. 


