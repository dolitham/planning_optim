import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
# nltk.download('stopwords')
from keras.preprocessing.image import ImageDataGenerator
plt.style.use('dark_background')

data = pd.read_csv('Flipkart/flipkart_com-ecommerce_sample_1050.csv', parse_dates=['crawl_timestamp'])
idx_images = [0, 100, 1000]

for idx_image in idx_images:
    id_image = data['image'].iloc[idx_image]
    filename_image = 'Flipkart/Images/' + id_image
    pfx_save = 'processed/' + id_image + '__'

    img = cv2.cvtColor(cv2.imread(filename_image), cv2.COLOR_BGR2GRAY).astype(np.uint8)

    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_img = cv2.filter2D(src=img, ddepth=-1, kernel=sharpening_kernel)
    cv2.imwrite(pfx_save + 'sharp_and_original_image.jpg', np.hstack((img, sharp_img)))

    equalized_img = cv2.equalizeHist(img)
    cv2.imwrite(pfx_save + 'equalized_and_original_image.jpg', np.hstack((img, equalized_img)))

    equalized_sharp_img = cv2.equalizeHist(sharp_img)
    sharp_equalized_img = cv2.filter2D(src=equalized_img, ddepth=-1, kernel=sharpening_kernel)
    cv2.imwrite(pfx_save + 'eq_sharp_vs_sharp_eq.jpg', np.hstack((equalized_sharp_img, sharp_equalized_img)))


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)
datagen.fit(img)
