import data_processing as dp
import numpy as np
import torch
from torch.utils.data import Dataset

class PrivacyPoliciesDataset(Dataset):  
    
    def __init__(self, folder, path, word2idx, labels, read = False):
        
        self.folder = folder
       
        self.path = path
        
        self.segments_list, self.labels_list = self.unpack_segments(self.folder, word2idx, labels, read)
        
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
    
        same_size = True
        
        maximun_len = 0
              
        segments_iterator = iter(self.segments_list)
        
        prev_length = len(segments_iterator.next())

        for segment in segments_iterator:

            same_size = same_size and (len(segment) == prev_length)
            
            if len(segment) > maximun_len:

                maximun_len = len(segment)
                
            prev_length = len(segment)            

        if same_size:
            
            print("All segments already have the same size. Size: " + str(maximun_len))
            
        else:
            
            segments_length = clearance + maximun_len
            
            print("Resizing segments (filling with zeros). Target size: " + str(segments_length))
        
            for i, segment in enumerate(self.segments_list):

                array = segment.numpy()

                zeros_to_prepend = (segments_length - len(array))/2

                zeros_to_append = segments_length - len(array) - zeros_to_prepend

                resized_array = np.append(np.zeros(zeros_to_prepend), array)

                resized_array = np.append(resized_array, np.zeros(zeros_to_append))

                self.segments_list[i] = torch.tensor(resized_array, dtype = torch.int64)
                
    def expand_dimensions(self):
        
        """
        
        This method transforms all the 1-dimensional tensors inside segments_list to 2-dimensional tensor. This is necessary
        before calling group_samples and before using the dataset in a CNN. 
        
        """
        
        for i, segment in enumerate(self.segments_list):
            
            self.segments_list[i] = segment.unsqueeze(0)
    
    def unpack_segments(self, folder, word2idx, labels, read):

        segments_list = []

        labels_list = []

        files_matrices, files_labels = dp.process_dataset(folder ,labels, word2idx, read)

        for segments_matrix in files_matrices:

            for segment_matrix in segments_matrix:

                segments_list.append(torch.tensor(segment_matrix))

        for file_labels in files_labels:

            for segment_label in file_labels:

                labels_list.append(torch.tensor(segment_label))

        return (segments_list, labels_list) 
    
    def group_samples(self):
        
        same_size = True
        
        segments_iterator = iter(self.segments_list)
        
        prev_length = len(segments_iterator.next())

        for segment in segments_iterator:

            same_size = same_size and (len(segment) == prev_length)
                
            prev_length = len(segment)            

        if not same_size:
            
            print("Can't group samples into one Tensor. All samples must have the same size.")
            
            print("Call resize_segments on dataset.")
            
        else: 
            
            print("Grouping samples into one Tensor")
            
            self.segments_list = torch.stack(self.segments_list)
            
            self.labels_list = torch.stack(self.labels_list)
                   