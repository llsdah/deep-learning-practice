import numpy as np

class TextDataset:
    def __init__(self, texts, labels=None, transform=None):
        self.texts = texts
        self.labels = labels
        self.transform = transform
        if self.labels is not None:
            assert len(self.texts) == len(self.labels), '텍스트와 레이블의 길이가 일치하지 않습니다.'

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, 색인):
        sample = self.texts[색인]
        if self.transform:
            sample = self.transform(sample)
            
        if self.labels is not None:
            label = self.labels[색인]
            return sample, label
        return sample


def 시퀀스패딩(sample, 최대길이, 패딩값=0):
    assert len(sample) > 0, '샘플이 비어 있습니다.'
    sample = sample[:최대길이]
    return sample + [패딩값] * (최대길이 - len(sample))

def 배치병합(batch):
    samples, labels = zip(*batch)
    문서길이 = [len(sample) for sample in samples]
    최대길이 = max(문서길이) # 배치 내에서 가장 긴 샘플의 길이를 최대길이로 설정
    정규화된_샘플 = [시퀀스패딩(sample, 최대길이) for sample in samples]
    return np.array(정규화된_샘플), np.array(labels)