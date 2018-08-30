from string import lower
from os.path import isfile, join
from os import listdir
import nltk
import pickle
import numpy as np


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
    
    s_sentence = list()
    
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
            
            print("Warning: At least one token is not present in the word2idx dict. For instance: " + token + ". Not found: " 
                   + str(not_found))
        
    return s_sentence

def get_tokens(path, read = False):
    """
    
    Checks all the files in filespath and returns all a set of all the words found in the files. The function will
    ignore all the folders inside filespath automatically. We set all the words to be lower case. The function will
    check if the a file with all the tokens is available. In that case this function will be much faster. 
    
    Args:
        filespath: string, path to the folder with all the files containing the words that we want to extract.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        dictionary: set, set containing all the different words found in the files. 
    
    """

    if isfile("dictionary.pkl") and read == True:
        
        print("Loading from file dictionary.pkl")
        
        input_file = open("dictionary.pkl","rb")
        
        dictionary = pickle.load(input_file)
        
        input_file.close()
        
    else:
        
        print("Processing dataset ...")
    
        dictionary = set()

        files = [f for f in listdir(path) if isfile(join(path, f))]

        for f in files:

            file_path = join(path, f)

            opened_file = open(file_path,'r')

            for i, line in enumerate(opened_file):

                a = line.split('","')

                a[1] = map(lower,set(nltk.word_tokenize(a[1])))

                dictionary = dictionary.union(a[1])

        output_file = open("dictionary.pkl","wb")

        pickle.dump(dictionary, output_file)

        output_file.close()
            
    return dictionary

def label_to_vector(label, labels): 
    """
    
    Returns a vector representing the label passed as an input.
    
    Args:
        label: string, label that we want to transform into a vector.
        labels: dictionary, dictionary with the labels as the keys and indexes as the values.
    Returns:
        vector: np.array, 1-D array of lenght 10.
        
    """
    
    vector = np.zeros((10))
    
    index = labels[label]
    
    vector[index] = 1
    
    return vector

def get_glove_dicts(path, dims, read = False):
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
    
    if isfile("word2vector.pkl") and isfile("word2idx.pkl") and read == True:
        
        print("Loading from files word2vector.pkl and word2idx.pkl")

        input_file1 = open("word2vector.pkl","rb")
        
        word2vector = pickle.load(input_file1)
        
        input_file1.close()

        input_file2 = open("word2idx.pkl","rb")
        
        word2idx = pickle.load(input_file2)
        
        input_file2.close()

    else:
        
        print("Processing dataset ...")

        words = []

        word2idx = {}

        idx = 0

        vectors = []

        with open(join(path,path + "." + str(dims) + "d.txt")) as glove_file:

            for line in glove_file:

                split_line = line.split()

                word = split_line[0]

                words.append(word)

                word2idx[word] = idx

                idx += 1

                vector = np.array(split_line[1:]).astype(np.float)

                vectors.append(vector)
        
        word2vector = {w: vectors[word2idx[w]] for w in words}

        output_file1 = open("word2vector.pkl","wb")

        pickle.dump(word2vector, output_file1)

        output_file1.close()

        output_file2 = open("word2idx.pkl","wb")

        pickle.dump(word2idx, output_file2)

        output_file2.close()
    
    return (word2vector, word2idx)

def get_weight_matrix(dictionary, word2vector, dims, read = False):
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
        word2idx: dictionary, the keys are the words and the values the index where we can find the vector in weights_matrix

    """

    if isfile("weights_matrix.pkl") and read == True:
        
        print("Loading from file weights_matrix.pkl")

        input_file1 = open("weights_matrix.pkl","rb")
        
        weights_matrix = pickle.load(input_file1)
        
        input_file1.close()
        
        input_file2 = open("word2idx.pkl","rb")
        
        word2idx = pickle.load(input_file2)
        
        input_file2.close()

    else:
        
        print("Processing dataset ...")

        matrix_len = len(dictionary)

        weights_matrix = np.zeros((matrix_len, dims))
        
        word2idx = {}

        words_found = 0

        for i, word in enumerate(dictionary):

            try: 

                weights_matrix[i] = word2vector[word]
                
                word2idx[word] = i

                words_found += 1

            except KeyError:

                weights_matrix[i] = np.random.normal(scale=0.6, size=(dims, ))
                
                word2idx[word] = i

        output_file1 = open("weights_matrix.pkl","wb")

        pickle.dump(weights_matrix, output_file1)

        output_file1.close()
        
        output_file2 = open("word2idx.pkl","wb")

        pickle.dump(word2idx, output_file2)
        
        output_file2.close()
            
    return (weights_matrix, word2idx)

def process_dataset(path, word2idx, labels, read = False):
    """
    
    This function process all the privacy policy files and transforms all the segments into lists of integers. It also 
    transforms all the labels into a list of 0s except in one position where we will find a 1. It will also place .pkl files 
    into the processed_data folder so that we can load the data from there instead of having to process the whole dataset.
    
    Args:
        path: string, path where all the files we want to process are located (all the privacy policies).
        word2idx: dictionary the keys are the words and the values the index where we can find the vector in weights_matrix.
        labels: labels: dictionary, dictionary with the labels as the keys and indexes as the values.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not. 
    Returns:
        sentence_matrices: list, a list of lists of lists containing the segments of the files transformed into integers. 
        sentence_matrices[i][j][k] -> "i" is for the file, "j" for the line and "k" for the token. 
        labels_matrices: list, a list of lists of lists containing the labels of the dataset. labels_matrices[i][j][k] -> "i"
        is for the file, "j" for the line and "k" for the boolean variable specifying the presence of the a label.
    
    """
    
    files = [f for f in listdir(path) if isfile(join(path, f))]
    
    files.remove(".keep")
    
    all_files = True
    
    for f in files:
        
        path_sentence_matrices = "processed_data/sentence_matrices/" + f.replace(".csv","_sentence_matrix.pkl")
        
        path_labels_matrices = "processed_data/labels_matrices/" + f.replace(".csv","_labels_matrix.pkl")
        
        all_files = all_files and isfile(path_sentence_matrices) and isfile(path_labels_matrices)
    
    if all_files == True and read == True:
        
        print("Loading from folders processed_data/sentence_matrices/ and processed_data/labels_matrices/")
        
        sentence_matrices = list()

        labels_matrices = list()

        for f in files:
            
            path_sentence_matrices = "processed_data/sentence_matrices/" + f.replace(".csv","_sentence_matrix.pkl")
        
            path_labels_matrices = "processed_data/labels_matrices/" + f.replace(".csv","_labels_matrix.pkl")
            
            input_file1 = open(path_sentence_matrices,"rb")
            
            f_sentence_matrix = pickle.load(input_file1)
            
            input_file1.close()
            
            input_file2 = open(path_sentence_matrices,"rb")
            
            f_labels_matrix = pickle.load(input_file2)
            
            input_file2.close()
            
            sentence_matrices.append(f_sentence_matrix)
            
            labels_matrices.append(f_labels_matrix)
        
    else:
        
        print("Processing dataset ...")
        
        sentence_matrices = list()

        labels_matrices = list()

        for f in files:   

            file_path = join(path,f)

            sentence_matrix = list()

            labels_matrix = list()

            opened_file = open(file_path,'r')

            for line in opened_file:

                split_line = line.split('","')

                sentence = split_line[1]

                label = split_line[2].replace('"',"").replace('\n',"")

                sentence_matrix.append(sentence_serialization(sentence, word2idx))

                labels_matrix.append(label_to_vector(label, labels))
                
                path_sentence_matrices = "processed_data/sentence_matrices/" + f.replace(".csv","_sentence_matrix.pkl")
        
                path_labels_matrices = "processed_data/labels_matrices/" + f.replace(".csv","_labels_matrix.pkl")

                f1 = open(path_sentence_matrices, "wb")

                pickle.dump(sentence_matrix, f1)

                f1.close()

                f2 = open(path_labels_matrices, "wb")

                pickle.dump(labels_matrix, f2)

                f2.close()

            sentence_matrices.append(sentence_matrix)

            labels_matrices.append(labels_matrix)
                    
    return (sentence_matrices, labels_matrices) 
