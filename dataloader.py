import numpy as np
import os
import heartpy as hp
import copy
from scipy.signal import stft
import pywt
import torch
import random
import torchvision.transforms as transforms
from utils import normalize, beat_normalize
from torch.utils.data import Dataset

class TrainSet(Dataset):
    def __init__(self, processed_data):
        """
        Args:
            processed_data (dict): 
            전처리된 데이터가 포함된 사전. 각 항목은 파일명에 해당하며,
            값은 각 청크에 대한 데이터가 담긴 리스트입니다.
                                    
            {
                'chunk_index': idx,
                'time': time,
                'normalized_hr': normalized_signal_h,
                'reconstructed_signal': normalized_reconstructed_signal,
                'peak_hr': normalized_peak_h, 
            }
        """
        self.train_data = self._prepare_data(processed_data)
        
    def _prepare_data(self, processed_data):
        
        all_samples = []
        for file_name, file_data in processed_data.items():
            for data in file_data:
                
                normalized_hr = data['normalized_hr']
                reconstructed_signal = data['reconstructed_signal']
                peak_hr = data['peak_hr']
                
                # 데이터가 (time_steps, channels) 형태가 되도록 결합
                combined_chunk = np.stack([normalized_hr, reconstructed_signal, peak_hr], axis=-1) # (560, 3)
                
                # 청크의 형태가 (350, 3)인 경우에만 리스트에 추가
                if combined_chunk.shape == (560, 3):
                    all_samples.append(combined_chunk)
                
                else: print("len: ", combined_chunk.shape)
        
        # 리스트를 numpy 배열로 변환
        all_samples = np.array(all_samples, dtype=np.float32)
        
        return all_samples
    
    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, index):
        time_instance = self.train_data[index]

        # Short Time Fast Fourier Transform
        # 500 is the sample rate of PTB-XL, 360 is the sample rate of MIT-BIH
        f,t, Zxx = stft(time_instance.transpose(1,0),fs=100, window='hann',nperseg=125)
        spectrogram_instance = np.abs(Zxx)  #(12, 63, 78)
        spectrogram_instance = spectrogram_instance.transpose(1,2,0)     #(63, 78, 12)
        return time_instance, spectrogram_instance


class Testset(Dataset):
    def __init__(self, processed_data):
        """
        Args:
            processed_data (dict): 
            전처리된 데이터가 포함된 사전. 각 항목은 파일명에 해당하며,
            값은 각 청크에 대한 데이터가 담긴 리스트입니다.
                                    
            {
                'file_name': file_name,
                'chunk_index': idx,
                'time': time,
                'normalized_hr': normalized_signal_h,
                'reconstructed_signal': normalized_reconstructed_signal,
                'peak_hr': normalized_peak_h, 
            } 
        """
        self.data, self.time = self._prepare_data(processed_data)
        
    def _prepare_data(self, processed_data):
        """
        전처리된 데이터를 모델 입력 형태로 변환합니다.
        """
        all_samples = []
        all_whens = []
        
        for file_name, file_data in processed_data.items():
            for data in file_data:
                normalized_hr = data['normalized_hr']
                reconstructed_signal = data['reconstructed_signal']
                peak_hr = data['peak_hr']
                
                filename = data['file_name']
                chunk_index = data['chunk_index']
                timestamps = data['time']  # 타임스탬프는 문자열로 저장되어 있음
                
                # filename과 chunk_index를 timestamps와 동일한 길이로 확장
                filename_array = np.repeat(filename, len(timestamps))
                chunk_index_array = np.repeat(chunk_index, len(timestamps))
                
                # 데이터가 (time_steps, channels) 형태가 되도록 결합
                combined_chunk = np.stack([normalized_hr, reconstructed_signal, peak_hr], axis=-1) # (560, 3)
                combined_chunk_t = np.stack([filename_array, chunk_index_array, timestamps], axis=-1) # (560, 3)
                
                if combined_chunk.shape == (560, 3) and combined_chunk_t.shape == (560, 3):
                    all_samples.append(combined_chunk)
                    all_whens.append(combined_chunk_t)
                else:
                    print(f"Skipping due to shape mismatch: {combined_chunk.shape}, {combined_chunk_t.shape}")
        
        all_samples = np.array(all_samples, dtype=np.float32)
        all_whens = np.array(all_whens, dtype=object)  # 문자열을 포함하므로 dtype을 object로 설정
        
        return all_samples, all_whens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 데이터 인덱스
        Returns:
            torch.Tensor: (time_steps, channels) 형태의 텐서
        """
        time_instance = self.data[idx]
        timestamp = self.time[idx]
        # Short Time Fast Fourier Transform
        f,t, Zxx = stft(time_instance.transpose(1,0),fs=100, window='hann',nperseg=125)
        spec_instance = np.abs(Zxx)  #(12, 63, 78)
        spec_instance = spec_instance.transpose(1,2,0)  
        return torch.tensor(time_instance, dtype=torch.float32), torch.tensor(spec_instance, dtype=torch.float32), timestamp
