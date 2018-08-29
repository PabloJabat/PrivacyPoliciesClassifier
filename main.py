import data_processing as dp
import sys
import random
import pickle

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

if __name__ == "__main__":

	path = sys.argv[1]

	dims = sys.argv[2]

	labels_file = open("labels.pkl","rb")

	labels = pickle.load(labels_file)

	labels_file.close()
    
	tokens = dp.get_tokens("raw_data")
    
	print("Random samples: " + str(random.sample(tokens, 2)))

	word2vector, word2idx = dp.get_glove_dicts(path, dims, True)

	weights_matrix, word2idx = dp.get_weight_matrix(tokens, word2vector, dims, True)

	sentence_matrices, labels_matrices = dp.process_dataset(path, word2idx, labels, read = False)

	print("Program executed succesfully ...")
