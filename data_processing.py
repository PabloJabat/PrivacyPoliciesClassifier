from string import lower
from os.path import isfile
import nltk
import pickle


def sentence_serialization(sentence, word2idx, lower_case = True):
    """ 
    
    Transforms a sentence into a list of integers.
    
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
    
    if lower_case: 
        
        tokens = map(lower,nltk.word_tokenize(sentence))  
        
    else:
        
        tokens = nltk.word_tokenize(sentence)
    
    for token in tokens:       
        
        try:
            
            s_sentence.append(word2idx[token])
            
        except KeyError:
            
            print("Warning: At least one token is not present in the word2idx dict")
        
    return s_sentence

def get_tokens(filespath):
    """
    
    Checks all the files in filespath and returns all a set of all the words found in the files. The function will
    ignore all the folders inside filespath automatically. We set all the words to be lower case. The function will
    check if the a file with all the tokens is available. In that case this function will be much faster. 
    
    Args:
        filespath: string, path to the folder with all the files containing the words that we want to extract.
    Returns:
        dictionary: set, set containing all the different words found in the files. 
    
    """

    if isfile("dictionary.pkl"):
        
        input_file = open("dictionary.pkl","rb")
        
        dictionary = pickle.load(input_file)
        
        input_file.close()
        
    else:
    
        dictionary = set()

        files = [f for f in listdir(filespath) if isfile(join(filespath, f))]

        for f in files:

            filepath = join(filespath, f)

            openedfile = open(filepath,'r')

            for i, line in enumerate(openedfile):

                a = line.split('","')

                a[1] = map(lower,set(nltk.word_tokenize(a[1])))

                dictionary = dictionary.union(a[1])

        output_file = open("dictionary.pkl","wb")

        pickle.dump(dictionary, output_file)

        output_file.close()
            
    return dictionary