import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from bcgpy import separate_breath, separate_heart, normalize, calculate_peaks

class BCGDataset(Dataset):
    def __init__(self, csv_file, SR, transform=None):
        """
        Args:
            csv_file (string): CSV 파일 경로.
            SR (int): 샘플링 레이트.
            transform (callable, optional): 데이터에 적용할 변환 함수.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.SR = SR  # 샘플링 레이트
        self.transform = transform

        # BCG 신호 열 추출
        self.bcgs = self.data_frame['bcg'].apply(lambda x: np.array(x.split(','), dtype=float)).tolist()

    def __len__(self):
        return len(self.bcgs)

    def __getitem__(self, idx):
        # 원본 BCG 신호 불러오기
        bcg_signal = self.bcgs[idx]

        # 전처리 적용
        breath = separate_breath(bcg_signal, self.SR)
        heart = separate_heart(bcg_signal, self.SR)
        normalized_heart = normalize(heart)
        normalized_breath = normalize(breath)
        peaks = calculate_peaks(normalized_heart)


        sample = {
            'original': torch.tensor(bcg_signal, dtype=torch.float32).unsqueeze(0),
            'breath_component': torch.tensor(breath, dtype=torch.float32).unsqueeze(0),
            'heart_component': torch.tensor(heart, dtype=torch.float32).unsqueeze(0),
            'normalized_heart_signal': torch.tensor(normalized_heart, dtype=torch.float32).unsqueeze(0),
            'normalized_breath_signal': torch.tensor(normalized_breath, dtype=torch.float32).unsqueeze(0),
            'peaks': torch.tensor(peaks, dtype=torch.float32).unsqueeze(0)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def load_data(csv_file, SR, batch_size, shuffle=True):
    dataset = BCGDataset(csv_file=csv_file, SR=SR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
