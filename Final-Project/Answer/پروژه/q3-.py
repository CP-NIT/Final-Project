import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# خواندن فایل تصویر و تبدیل آن به یک آرایه دو بعدی
img = plt.imread('image.jpg')
img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])  # تبدیل به خاکستری

# ایجاد تابعی که یک آرایه دو بعدی به عنوان تصویر و یک آرایه دو بعدی دیگر به عنوان فیلتر می‌گیرد، 
# کانولوشن دو بعدی را بر روی این دو انجام می‌دهد و نتیجه این عملیات را برمی‌گرداند
def convolution(image, filter):
    (iH, iW) = image.shape
    (fH, fW) = filter.shape
    pad = (fW - 1) // 2
    image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')  # پدینگ تصحیح شده
    output = np.zeros((iH, iW), dtype="float32")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * filter).sum()
            output[y - pad, x - pad] = k
    return output

# یافتن فیلتر مناسب برای تشخیص لبه
filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# انجام عملیات کانولوشن بر روی تصویر و فیلتر، سپس نمایش نتیجه در خروجی
result = convolution(img, filter)

# نمایش تصویر اصلی و نتیجه تشخیص لبه
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('تصویر اصلی'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(result, cmap='gray', vmin=0, vmax=255)  # اضافه کردن vmin و vmax برای مقیاس تصویر
plt.title('تشخیص لبه'), plt.xticks([]), plt.yticks([])
plt.show()