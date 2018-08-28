import data_processing as dp

if __name__ == "__main__":

	print("this is the main")

	sentence = "This is a sentence"

	word2idx = {"this": 0, "is": 1, "a": 2, "sentence": 3}

	s_sentence = dp.sentence_serialization(sentence, word2idx)

	print(s_sentence)
