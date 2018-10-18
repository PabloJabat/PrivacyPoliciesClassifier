from string import lower
from os.path import isfile, join
from os import listdir
import nltk
import pickle
import numpy as np
import pandas as pd

def get_number_segments(folder):
    """
    
    Computes the number of segments in the files inside the folder
    
    Args:
        folder: string, path to the folder we want to examine
    Returns:
        number_segments: integer, number of segments inside the folder
    
    """
    
    number_segments = 0 
    
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    
    for f in files:

        with open(f, "rb") as f_opened:

            number_segments += len(pickle.load(f_opened))
            
    return number_segments
            
def sentence_serialization(sentence, word2idx, lower_case = True):
    """ 
    
    Transforms a sentence into a list of integers. No integer will be appended if the token is not present in word2idx.
    
    Args:
        sentence: string, sentence that we want to serialize.
        word2idx: dictionary, dictionary with words as keys and indexes as values.
        lower_case: boolean, turns all words in the sentence to lower case. Useful if word2idx 
        doesn't support upper case words.
    Returns: 
        s_sentence: list, list containing the indexes of the words present in the sentence. 
        s_sentence stands for serialized sentence.
        
    """
    
    s_sentence = []
    
    not_found = 0
    
    if lower_case: 
        
        tokens = map(lower,nltk.word_tokenize(sentence))  
        
    else:
        
        tokens = nltk.word_tokenize(sentence)
    
    for token in tokens:       
        
        try:
            
            s_sentence.append(word2idx[token])
            
        except KeyError:
            
            not_found += 1
            
            print("Warning: At least one token is not present in the word2idx dict. For instance: " + token + 
                  
                  ". Not found: " + str(not_found))
        
    return s_sentence

def get_tokens(inputh_path, output_path, read = False):
    """
    
    Checks all the files in filespath and returns a set of all the words found in the files. The path has to have tow 
    folders, one called train with all the files meant for training the NN and another called test with all the files that
    will be used for testing. The function will ignore all the folders inside filespath automatically. We set all the words 
    to be lower case. The function will check if the a file with all the tokens is available. In that case this function 
    will be much faster. 
    
    Args:
        filespath: string, path to the folder with all the files containing the words that we want to extract.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        dictionary: set, set containing all the different words found in the files. 
    
    """

    dictionary_path = join(output_path, "dictionary.pkl")
    
    if isfile(dictionary_path) and read:
        
        print("Loading from file dictionary.pkl")
        
        with open(dictionary_path,"rb") as dictionary_file:
        
            dictionary = pickle.load(dictionary_file)
        
    else:
        
        print("Processing dataset ...")
    
        dictionary = set()

        files = [join(inputh_path, f) for f in listdir(inputh_path) if isfile(join(inputh_path, f))]
        
        files.remove(join(inputh_path,".keep"))
              
        for f in files:

            opened_file = open(f,'r')

            for i, line in enumerate(opened_file):

                a = line.split('","')

                a[1] = map(lower,set(nltk.word_tokenize(a[1])))

                dictionary = dictionary.union(a[1])

        with open(dictionary_path, "wb") as dictionary_file:

            pickle.dump(dictionary, dictionary_file)
            
    return dictionary

def label_to_vector(label, labels): 
    """
    
    Returns a vector representing the label passed as an input.
    
    Args:
        label: string, label that we want to transform into a vector.
        labels: dictionary, dictionary with the labels as the keys and indexes as the values.
    Returns:
        vector: np.array, 1-D array of lenght 9.
        
    """
    
    vector = np.zeros((9))
    
    try:
    
        index = labels[label]
    
        vector[index] = 1
        
    except KeyError:
        
        vector = np.zeros((9))
    
    return vector

def get_glove_dicts(inputh_path, output_path, dims, read = False):
    """
    
    This functions returns two dictionaries that process the glove.6B folder and gets the pretrained 
    embedding vectors.
    
    Args:
        path: string, path to the folder containing the glove embeddings
        dims: integer, embeddings dimensionality to use.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        word2vector: dictionary, the keys are the words an the values are the embeddings associated with that word.
        word2idx: dictionary, the keys are the words and the values are the indexes associated with that word.
    
    """
    
    word2vector_path = "word2vector_globe_" + str(dims) + ".pkl"
    
    word2vector_path = join(output_path, word2vector_path)
    
    word2idx_path = "word2idx_globe_" + str(dims) + ".pkl"
    
    word2idx_path = join(output_path, word2idx_path)
    
    if isfile(word2vector_path) and isfile(word2idx_path) and read:
        
        print("Loading from files word2vector_globe_{0}.pkl and word2idx_globe_{0}.pkl".format(dims))

        with open(word2vector_path,"rb") as word2vector_file:
        
            word2vector = pickle.load(word2vector_file)

    else:
        
        print("Processing dataset ...")

        words = [None]

        word2idx = {None: 0}

        idx = 1

        vectors = [np.zeros(dims)]

        with open(join(inputh_path, inputh_path + "." + str(dims) + "d.txt")) as glove_file:

            for line in glove_file:

                split_line = line.split()

                word = split_line[0]

                words.append(word)

                word2idx[word] = idx

                vector = np.array(split_line[1:]).astype(np.float)

                vectors.append(vector)
                
                idx += 1
        
        word2vector = {w: vectors[word2idx[w]] for w in words}
        
        with open(word2vector_path,"wb") as word2vector_file:
        
            pickle.dump(word2vector, word2vector_file)

    return word2vector

def get_weight_matrix(dictionary, word2vector, dims, output_path, read = False):
    """

    This function returns a matrix containing the weights that will be used as pretrained embeddings. It will read 
    weights_matrix.pkl file as long as it exists. This will make the code much faster. 

    Args:
        dictionary: set, set containing all the words present in the dataset.
        word2vector: dictionary, the keys are the words and the values are the embeddings.
        dims: integer, dimensionality of the embeddings.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        weights_matrix: np.array, matrix containing all the embeddings.
        word2idx: dictionary, the keys are the words and the values the index where we can find the vector in 
        weights_matrix

    """
    
    weights_path = "weights_matrix_" + str(dims) + ".pkl"
    
    weights_path = join(output_path, weights_path)
    
    word2idx_path = "word2idx_" + str(dims) + ".pkl"
    
    word2idx_path = join(output_path, word2idx_path)

    if isfile(weights_path) and isfile(word2idx_path) and read:
        
        print("Loading from file weights_matrix.pkl")

        with open(weights_path,"rb") as weights_file:
        
            weights_matrix = pickle.load(weights_file)

        with open(word2idx_path,"rb") as word2idx_file:
        
            word2idx = pickle.load(word2idx_file)

    else:
        
        print("Processing dataset ...")
        
        # We add 1 to onclude the None value
        
        matrix_len = len(dictionary) + 1

        weights_matrix = np.zeros((matrix_len, dims))
        
        word2idx = {None: 0}

        words_found = 0

        for i, word in enumerate(dictionary,1):

            try: 

                weights_matrix[i] = word2vector[word]
                
                word2idx[word] = i

                words_found += 1

            except KeyError:

                weights_matrix[i] = np.random.normal(scale=0.6, size=(dims, ))
                
                word2idx[word] = i

        with open(weights_path,"wb") as weights_file:

            pickle.dump(weights_matrix, weights_file)
            
        with open(word2idx_path,"wb") as word2idx_file:

            pickle.dump(word2idx, word2idx_file)
            
    return (weights_matrix, word2idx)

def process_dataset(labels, word2idx, read = False):
    """
    
    This function process all the privacy policy files and transforms all the segments into lists of integers. It also 
    transforms all the labels into a list of 0s except in the positions associated with the labels in which we will find 1s
    where we will find a 1. It will also place .pkl files into the processed_data folder so that we can load the data from 
    there instead of having to process the whole dataset.
    
    Args:
        path: string, path where all the files we want to process are located (all the privacy policies).
        word2idx: dictionary the keys are the words and the values the index where we can find the vector in 
        weights_matrix.
        labels: labels: dictionary, dictionary with the labels as the keys and indexes as the values.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not. 
    Returns:
        sentence_matrices: list, a list of lists of lists containing the segments of the files transformed into integers. 
        sentence_matrices[i][j][k] -> "i" is for the file, "j" for the line and "k" for the token. 
        labels_matrices: list, a list of lists of lists containing the labels of the dataset. labels_matrices[i][j][k] ->
        "i" is for the file, "j" for the line and "k" for the boolean variable specifying 
        the presence of the a label.
    """
    
    
    """
    
    Helper functions
    
    """
    
    def pickle_matrix(matrix, path):
        
        with open(path,"wb") as output_file:

            pickle.dump(matrix, output_file)
    
    def unpickle_matrix(path):
        
        with open(path,"rb") as input_file:

            matrix = pickle.load(input_file)
        
        return matrix
    
    """
    
    main code of process_dataset
    
    """
    
    input_path = "agg_data"
    
    output_path = "processed_data"
    
    path_sentence_matrices = join(output_path, "all_sentence_matrices.pkl")

    path_labels_matrices = join(output_path, "all_label_matrices.pkl")
    
    if isfile(path_sentence_matrices) and isfile(path_labels_matrices) and read:
        
        print("Loading from " + path_sentence_matrices + " and " + path_labels_matrices)       
        
        sentence_matrices = unpickle_matrix(path_sentence_matrices)

        labels_matrices = unpickle_matrix(path_labels_matrices)

        return sentence_matrices, labels_matrices 
        
    else:
        
        print("Processing dataset ...")
        
        with open("agg_data/agg_data.pkl",'rb') as dataframe_file:

            opened_dataframe = pickle.load(dataframe_file)

        num_records = len(opened_dataframe)

        num_labels = len(opened_dataframe["label"].iloc[0])

        sentence_matrices = np.zeros(num_records, dtype = 'object')

        labels_matrices = np.zeros((num_records, num_labels))

        for index, row in opened_dataframe.iterrows():

            segment = row["segment"]

            label = row["label"]

            sentence_matrices[index] = sentence_serialization(segment, word2idx)

            labels_matrices[index] = label

        path_sentence_matrices = join(output_path, "all_sentence_matrices.pkl")

        path_labels_matrices = join(output_path, "all_label_matrices.pkl")

        pickle_matrix(sentence_matrices, path_sentence_matrices)

        pickle_matrix(labels_matrices, path_labels_matrices)

        return sentence_matrices, labels_matrices               

def aggregate_data(read = False):    
    """
    
    This function processes raw_data and aggregates all the segments labels. Places all the files in the agg_data folder. 
    
    Args:
        read: boolean, if set to true it will read the data from agg_data folder as long as all the files are found 
        inside the 
        folder.
    Returns:
        Nothing.
    
    """ 
    
    """
    
    Helper functions
    
    """
    
    def aggregate_files(input_path, output_path, labels_dict):
        
        files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        
        files.remove(".keep")
        
        all_results = pd.DataFrame({'label' : [], 'segment' : []})
        
        for f in files:

            data = pd.read_csv(join(input_path,f), names = ["idx","segment","label"])

            data['label'] = data['label'].apply(lambda x: label_to_vector(x, labels_dict))

            labels_data = data[['idx','label']]

            labels = labels_data.groupby("idx").sum()

            segments = data[['idx','segment']].set_index('idx').drop_duplicates()

            result = pd.merge(labels, segments, left_index = True, right_index = True)
                
            all_results = pd.concat([all_results, result])
                
        all_results.reset_index(drop=True, inplace=True)
            
        folder_output_path = "agg_data"

        with file(join(output_path, "agg_data.pkl"),"wb") as output_file:

            pickle.dump(all_results, output_file)         
        
    """
    
    main code of aggregate_data
    
    """
    
    input_path = "raw_data"
    
    output_path = "agg_data"
    
    with open("labels.pkl","rb") as labels_file:
        
        labels_dict = pickle.load(labels_file)
                
    file_exists = isfile(join(output_path, "agg_data.pkl"))
                                
    if file_exists and read:

        print("agg_data.pkl are already in agg_data/")

    else: 

        print("Processing dataset in one file ...")

        aggregate_files(input_path, output_path, labels_dict)
    
def get_absent_words(dictionary, word2vector):
    """
    
    This function check if the words inside dictionary are present in word2vector which is a dictionary coming from a word
    embedding.
    
    Args:
        dictionary: set, set containing strings of words
        word2vector: dictionary, the keys are the words and the values are the embeddings   
    Returns:
        absent_words: list, list containing all the words that weren't found in the word embeddings word2vector
    
    """

    absent_words = []

    for word in dictionary:

        try:

            word2vector[word]

        except KeyError:

            absent_words.append(word)
            
    return absent_words
