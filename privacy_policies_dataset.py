import data_processing as dp
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