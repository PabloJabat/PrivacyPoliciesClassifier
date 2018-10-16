import data_processing as dp

import numpy as np

import torch

from torch import tensor

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

                self.segments_list[i] = torch.tensor(resized_array, dtype = torch.long)
                
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

class PrivacyPoliciesDataset_all(Dataset):
    
    def __init__(self, segments_array, labels_list, labels):
        
        self.segments_array = segments_array
       
        self.labels_tensor = tensor(labels_list)
        
        self.labels = labels
        
    def __len__(self):
        
        if self.segments_array.shape[0] == self.labels_tensor.shape[0]: 
            
            return self.segments_array.shape[0]
        
        else:
            
            print("Warning: number of segments don't match number of annotations")
            
            return self.segments_array.shape[0]     
    
    def __getitem__(self, idx):
        
        segment = self.segments_array[idx]
        
        label = self.labels_tensor[idx]
        
        return (segment, label)
    
    def split_dataset_randomly(self, ratio = 0.1):
    
        """

        This function randomly splits the dataset in two parts using the split ratio provided

        Args:
            dataset: torch.utils.data.Dataset, dataset containing the data to split
            ratio: double, percentage of data that will be retrieved inside s_dataset
        Returns:
            s_dataset: torch.utils.data.Dataset, dataset with length = len(dataset) * ratio
            b_dataset: torch.utils.data.Dataset, dataset with length = len(dataset) * (1 - ratio)

        """

        from random import sample

        labels = self.labels

        num_samples = int(ratio * len(self))

        s_dataset_idx_set = set(sample(range(len(self)), num_samples))

        b_dataset_idx_set = set(range(len(self))).difference(s_dataset_idx_set)

        s_dataset_idx_tensor = tensor(list(s_dataset_idx_set))

        b_dataset_idx_tensor = tensor(list(b_dataset_idx_set))

        s_dataset_data = self[s_dataset_idx_tensor]

        b_dataset_data = self[b_dataset_idx_tensor]

        s_dataset = PrivacyPoliciesDataset_all(s_dataset_data[0], s_dataset_data[1], labels)

        b_dataset = PrivacyPoliciesDataset_all(b_dataset_data[0], b_dataset_data[1], labels)

        return s_dataset, b_dataset
    
    def pickle_dataset(self, path):

        import pickle

        with open(path, "wb") as dataset_file:

            pickle.dump(self, dataset_file)
    
    @staticmethod
    def unpickle_dataset(path):
    
        import pickle

        with open(path, "rb") as dataset_file:

            dataset = pickle.load(dataset_file)

            return dataset
    
    @staticmethod
    def collate_data(batch):
        
        def stack_segments(segments, clearance = 2):

            import numpy as np

            segments_len = map(len, segments)

            max_len = max(segments_len)

            segments_list = []

            output_len = max_len + clearance * 2

            for i, segment in enumerate(segments):

                segment_array = np.array(segment)

                zeros_to_prepend = (output_len - len(segment_array))/2

                zeros_to_append = output_len - len(segment_array) - zeros_to_prepend

                resized_array = np.append(np.zeros(zeros_to_prepend), segment_array)

                resized_array = np.append(resized_array, np.zeros(zeros_to_append))

                segments_list.append(torch.tensor(resized_array, dtype = torch.int64))

                segments_tensor = torch.stack(segments_list).unsqueeze(1)

            return segments_tensor                         

        segments = [item[0] for item in batch]

        labels = [item[1] for item in batch]

        segments_tensor = stack_segments(segments)

        labels_tensor = torch.stack(labels)

        return [segments_tensor, labels_tensor]