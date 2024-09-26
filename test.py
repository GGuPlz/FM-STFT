import torch
import matplotlib.pyplot as plt

file_path = r'indoor_day1_4MHz_4096_test.pt'
length = int(512)
overlap = 0.75
data = torch.load(file_path)
signal = data[0]
signal = signal[0]
signal = signal.T
signal = signal[0, :] + 1j*signal[1, :]
spex = torch.stft(signal, n_fft=length, hop_length=int(length*(1-overlap)), window=torch.hann_window(length), return_complex=False, onesided=False)

print(spex.size())