import numpy as np
import torch
class TextDataset:
    def __init__(self, texts, labels=None, transform=None):
        self.texts = texts
        self.transform = transform
        self.labels = labels
        if self.labels is not None:
            assert len(self.texts) == len(self.labels), '텍스트와 레이블의 길이가 일치하지 않습니다.'
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        sample = self.texts[index]
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            label = self.labels[index]
            return sample, label
        
        return sample
    

def sequence_padding(sample, max_len, padding_value=0):
    assert len(sample) > 0, '샘플이 비어 있습니다.'
    sample = sample[:max_len]
    return sample + [padding_value] * (max_len - len(sample))


def combine_batch(batch):
    samples, labels = zip(*batch)
    len_doc = [len(sample) for sample in samples]
    max_len_doc = max(len_doc)
    numerize_sample = [sequence_padding(sample, max_len_doc) for sample in samples]
    return np.array(numerize_sample), np.array(labels)