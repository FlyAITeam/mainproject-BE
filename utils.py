import os
import time
import copy
import math
import pywt
import torch
import random
import pandas as pd
import numpy as np
import heartpy as hp
import torch.nn.functional as F

import scipy.signal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


########## TSRNet ORIGINAL functions
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def normalize(X_train_ori):
    X_train = copy.deepcopy(X_train_ori)
    for count in range(X_train.shape[0]):
        for j in range(12):
            seq = X_train[count][:,j]
            X_train[count][:,j] = 2*(seq-seq.min())/(seq.max()-seq.min())-1
    return X_train


def beat_normalize(X_train_ori):
    X_train = copy.deepcopy(X_train_ori)
    for j in range(12):
        seq = X_train[:,j]
        X_train[:,j] = 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
    return X_train


########## Preprocessing functions
def get_bcg_respiration_signal(bcg: np.ndarray, SR: int) -> np.ndarray:
    bcg = bcg.flatten()
    filter_coeffs = scipy.signal.butter(5, 0.7, 'low', fs=SR)
    return scipy.signal.filtfilt(*filter_coeffs, bcg)

def get_bcg_heartrate_signal(bcg: np.ndarray, SR: float) -> np.ndarray:
    bcg = bcg.flatten()
    filter_coeffs = scipy.signal.butter(5, [5, 15], 'band', fs=SR)
    return scipy.signal.filtfilt(*filter_coeffs, bcg)

def normalize_signal_window(signal: np.ndarray, window_size: int = 70) -> np.ndarray:
    normalized_signal = np.zeros_like(signal)
    
    for i in range(len(signal)):
        start_index = max(0, i - window_size + 1)
        window = signal[start_index:i+1]
        min_val = np.min(window)
        max_val = np.max(window)
        if max_val != min_val:
            normalized_signal[i] = (signal[i] - min_val) / (max_val - min_val)
        else:
            normalized_signal[i] = 0.5
    return normalized_signal

def calculate_checked_values(signal: np.ndarray, window_size: int = 10, threshold: float = 0.75):
    checked_values = np.zeros_like(signal)
    max_min_diff = np.zeros_like(signal)
    upto = np.zeros_like(signal)

    for i in range(len(signal)):
        if i >= window_size:
            window = signal[i-window_size+1:i+1]
            max_val = np.max(window)
            min_val = np.min(window)
            max_min_diff[i] = max_val - min_val
            if max_min_diff[i] >= threshold:
                checked_values[i] = max_val
                if checked_values[i-1] == 0:
                    upto[i] = max_val

    return checked_values, max_min_diff, upto

def calculate_upto_result(upto: np.ndarray):
    peaks = np.where(upto > 0)[0]
    
    if len(peaks) == 0:
        return np.zeros_like(upto)

    intervals = np.diff(peaks)

    if len(intervals) == 0:
        half_avg_interval = 0
    else:
        half_avg_interval = np.mean(intervals) / 2

    result = np.zeros_like(upto)
    
    for i in range(len(peaks)):
        if i == 0:
            result[peaks[i]] = upto[peaks[i]]

        elif (i>0) and (intervals[i-1] >= half_avg_interval):
            result[peaks[i]] = upto[peaks[i]]            

    return result

def calculate_permin(result: np.ndarray, sr: int):
    peaks = np.where(result > 0)[0]
    intervals = np.diff(peaks) / sr
    bpm = 60.0 / np.mean(intervals)
    return bpm

def find_minima_and_calculate_rr(signal: np.ndarray, sampling_rate: int):
    minima, _ = find_peaks(-signal)
    num_inflection_points = len(minima)
    time_duration = len(signal) / sampling_rate
    breaths = num_inflection_points
    respiratory_rate = breaths / time_duration * 60
    
    return respiratory_rate, minima

def normalize_signal(signal):
    max_val = np.max(signal)
    min_val = np.min(signal)
    if max_val == min_val:
        return np.full_like(signal, 0.5)  # 모든 값을 0.5로 설정하거나, 적절한 다른 값을 사용
    else:
        return (signal - min_val) / (max_val - min_val)
    
    
def extract_bcg_data(file_path):
    try:
        data = pd.read_csv(file_path).values
        if data.size == 0:
            print(f"Warning: {file_path} is empty.")
            return None
        return data
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_path} is empty.")
        return None

def split_data(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def process_all_csv_files(folder_path, sampling_rate=100, chunk_size=560):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # 결과를 저장할 변수
    all_processed_data = {}
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_path}")
        
        # 데이터 불러오기
        data = extract_bcg_data(file_path)
        
        if data is None:
            continue
        
        chunks = split_data(data, chunk_size)  # 데이터를 560개 단위로 분할
        
        # 파일별로 청크 저장
        file_processed_data = []
        
        for idx, chunk in enumerate(chunks):
            
            time = chunk[:, 0]
            signal = chunk[:, 1]
            
            # 데이터 분리
            filtered_hr = get_bcg_heartrate_signal(signal, sampling_rate)
            filtered_rp = get_bcg_respiration_signal(signal, sampling_rate)
            
            ## 데이터 정규화
            normalized_signal_h = normalize_signal_window(filtered_hr)
            normalized_signal_r = normalize_signal_window(filtered_rp)
            
            # peak detection
            _, _, upto_h = calculate_checked_values(normalized_signal_h)
            peak_h = calculate_upto_result(upto_h)
            
            combined_signal = 0.9 * peak_h + 0.1 * normalized_signal_r
            coeffs = pywt.wavedec(combined_signal, 'db4', level=4)
            reconstructed_signal = pywt.waverec(coeffs, 'db4')
            smoothed_reconstructed_signal_gaussian = gaussian_filter1d(reconstructed_signal, sigma=2)
            normalized_reconstructed_signal = normalize_signal(smoothed_reconstructed_signal_gaussian)
            
            peak_h = np.array(peak_h, dtype=np.float64)
            smoothed_peak_h_gaussian = gaussian_filter1d(peak_h, sigma=4)
            normalized_peak_h = normalize_signal(smoothed_peak_h_gaussian)
            
            # 결과를 변수에 저장
            file_processed_data.append({
                'file_name': file_name,
                'chunk_index': idx,
                'time': time,
                'normalized_hr': normalized_signal_h,
                'reconstructed_signal': normalized_reconstructed_signal,
                'peak_hr': normalized_peak_h, 
            })
        
        # 파일별로 데이터 저장
        all_processed_data[file_name] = file_processed_data
    
    return all_processed_data