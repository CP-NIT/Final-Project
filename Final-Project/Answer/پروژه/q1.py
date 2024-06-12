import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import unittest

# تابع کانولوشن سیگنال با کرنل
def convolution(signal, kernel):
    signal_len = len(signal)
    kernel_len = len(kernel)
    conv_len = signal_len + kernel_len - 1
    result = np.zeros(conv_len)
    for i in range(conv_len):
        for j in range(kernel_len):
            if i - j >= 0 and i - j < signal_len:
                result[i] += signal[i - j] * kernel[j]
    return result

# تابع فیلتر پایین گذر با استفاده از فیلتر باترورث
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# تابع فیلتر بالا گذر با استفاده از فیلتر باترورث
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# تابع فیلتر پایین گذر سیگنال با استفاده از فیلتر باترورث
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# تابع فیلتر بالا گذر سیگنال با استفاده از فیلتر باترورث
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# پارامترهای سیگنال
fs = 100.0
t = np.arange(0, 10, 1/fs)
signal = np.cos(0.2 * np.pi * t) + np.cos(0.02 * np.pi * t)

# پارامترهای فیلتر
low_cutoff = 0.1
high_cutoff = 0.15

# فیلتر کردن سیگنال
low_passed = butter_lowpass_filter(signal, low_cutoff, fs)
high_passed = butter_highpass_filter(signal, high_cutoff, fs)

# نمایش سیگنال ها
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title('سیگنال ترکیبی')
plt.xlabel('زمان (ثانیه)')
plt.ylabel('امپدانس')

plt.subplot(3, 1, 2)
plt.plot(t, low_passed)
plt.title('سیگنال فرکانس پایین (فیلتر شده)')
plt.xlabel('زمان (ثانیه)')
plt.ylabel('امپدانس')

plt.subplot(3, 1, 3)
plt.plot(t, high_passed)
plt.title('سیگنال فرکانس بالا (فیلتر شده)')
plt.xlabel('زمان (ثانیه)')
plt.ylabel('امپدانس')

plt.tight_layout()
plt.savefig('signal_separation.png')
plt.show()

class TestSignalProcessing(unittest.TestCase):
    def test_convolution(self):
        signal = np.array([1, 2, 3])
        kernel = np.array([0, 1, 0.5])
        expected = np.array([0, 1, 2.5, 4.5])
        result = convolution(signal.tolist(), kernel.tolist())
        np.testing.assert_array_almost_equal(np.array(result), expected)

if __name__ == '__main__':
    unittest.main()