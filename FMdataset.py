import torch
import torch.utils
from torch.utils.data import Dataset
import matplotlib.pyplot as plt 

def trans(signal, length, overlap):
    #signal = signal.to(torch.device("cuda"))
    window_fn = torch.hann_window(length)
    #window_fn = window_fn.to(torch.device("cuda"))
    signal = signal[0]
    signal = signal.T
    signal = signal[0, :] + 1j*signal[1, :]
    signal = torch.stft(signal, n_fft=length, hop_length=int(length*(1-overlap)), window=window_fn, return_complex=False, onesided=False)
    
    signal = signal.permute(2,0,1)
    min_val = torch.min(signal[0])
    max_val = torch.max(signal[0])
    signal[0] = (signal[0]-min_val)/(max_val-min_val)
    min_val = torch.min(signal[1])
    max_val = torch.max(signal[1])
    signal[1] = (signal[1]-min_val)/(max_val-min_val)
    return signal

class FMDataset(Dataset):
    def __init__(self, signal_path, label_path, length, overlap):
        self.FM = torch.load(signal_path)
        self.label = torch.load(label_path)
        self.label =self.label - 1
        self.STFT = torch.stack([trans(self.FM[i], length, overlap) for i in range(self.FM.shape[0])])

    def __getitem__(self, index):
        return self.STFT[index], self.label[index]

    def __len__(self):
        return len(self.STFT)






if __name__=='__main__': 
    data = FMDataset(r'indoor_day1_4MHz_4096_test.pt', r'indoor_day1_4MHz_4096_test_label.pt', 128, 0.75)
    a, _=data[0]
    print(a)
    plt.imshow(a[0])
    plt.show()