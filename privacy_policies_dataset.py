import data_processing as dp
import numpy as np
import torch
from torch.utils.data import Dataset

class PrivacyPoliciesDataset(Dataset):  
    
    def __init__(self, path, word2idx, labels):
       
        self.path = path
        
        self.segments_list, self.labels_list = self.unpack_segments(word2idx, labels)
        
    def __len__(self):
        
        if len(self.segments_list) == len(self.labels_list): 
            
            return len(self.segments_list)
        
        else:
            
            print("Warning: number of segments don't match number of annotations")
            
            return len(self.segments_list)       
    
    def __getitem__(self, idx):
        
        segment = self.segments_list[idx]
        
        label = self.labels_list[idx]
        
        return (segment, label)
    
    def resize_segments(self, clearance = 10):
    
        maximun_len = 0

        for segment in self.segments_list:

            if len(segment) > maximun_len:

                maximun_len = len(segment)

        maximun_len += clearance

        for i, segment in enumerate(self.segments_list):

            array = segment.numpy()

            zeros_to_prepend = (maximun_len - len(array))/2

            zeros_to_append = maximun_len - len(array) - zeros_to_prepend

            resized_array = np.append(np.zeros(zeros_to_prepend), array)

            resized_array = np.append(resized_array, np.zeros(zeros_to_append))

            self.segments_list[i] = resized_array
    
    def unpack_segments(self, word2idx, labels):

        segments_list = []

        labels_list = []

        files_matrices, files_labels = dp.process_dataset("raw_data", word2idx, labels)

        for segments_matrix in files_matrices:

            for segment_matrix in segments_matrix:

                segments_list.append(torch.tensor(segment_matrix))

        for file_labels in files_labels:

            for segment_label in file_labels:

                labels_list.append(segment_label)

        return (segments_list, labels_list) 