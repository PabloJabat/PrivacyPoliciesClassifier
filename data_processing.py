from string import lower
import nltk

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
