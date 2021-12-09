import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
import cv2
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

data = pd.read_csv('Flipkart/flipkart_com-ecommerce_sample_1050.csv', parse_dates=['crawl_timestamp'])
idx_image = random.randint(0, data.shape[0] - 1)
id_image = data['image'].iloc[idx_image]
filename_img = 'Flipkart/Images/' + id_image

img = load_img(filename_img).convert('L')
samples = np.expand_dims(img_to_array(img), 0)
datagen = ImageDataGenerator(rotation_range=30, zoom_range=[0.8, 1], horizontal_flip=True,
							 height_shift_range=[-0.15, 0.15], width_shift_range=[-0.15, 0.15],
							 fill_mode='nearest', rescale=.9)

exclude_indexes = [16,
 44,
 154,
 164,
 194,
 223,
 235,
 243,
 246,
 249,
 274,
 310,
 321,
 335,
 341,
 343,
 351,
 352,
 354,
 379,
 380,
 410,
 426,
 461,
 480,
 481,
 490,
 492,
 507,
 600,
 613,
 723,
 740,
 742,
 765,
 830,
 832,
 834,
 835,
 836,
 838,
 839,
 840,
 843,
 850,
 851,
 852,
 855,
 859,
 893,
 898,
 960,
 1002,
 1011]
exclude_kp_len = [101,
 49,
 100,
 137,
 129,
 118,
 83,
 54,
 101,
 83,
 135,
 36,
 145,
 128,
 121,
 0,
 117,
 143,
 2,
 146,
 57,
 87,
 82,
 126,
 123,
 123,
 123,
 135,
 135,
 115,
 53,
 140,
 60,
 94,
 119,
 64,
 119,
 32,
 32,
 72,
 74,
 115,
 63,
 74,
 59,
 134,
 134,
 128,
 121,
 15,
 147,
 145,
 70,
 103]

def get_sift_kp_and_descriptors(idx_image):
	id_image = data['image'].iloc[idx_image]
	filename_img = 'Flipkart/Images/' + id_image

	img = cv2.imread(filename_img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# noinspection PyUnresolvedReferences
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=150)
	return sift.detectAndCompute(gray, None), img, gray

def plot_image_w_keypoints(img, kp, gray):
	img_w_keypoints = cv2.drawKeypoints(gray, kp, img,
										flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	plt.imshow(img_w_keypoints, cmap='gray')
	plt.xticks([], "")
	plt.yticks([], "")
	plt.show()


kp_lengths = []
descriptors_df = pd.DataFrame()

for idx_image in tqdm(range(data.shape[0])):
	(kp, descriptors), img, gray = get_sift_kp_and_descriptors(idx_image)

	kp_lengths.append(len(kp))
	if len(kp) < 150:
		plot_image_w_keypoints(img, kp, gray)
	else:
		descriptors_df[data['image'].iloc[idx_image]] = np.concatenate(descriptors[:150, :])

descriptors_df = descriptors_df.transpose()

data['first_category'] = data['product_category_tree'].str.split('\"').map(lambda x:x[1].split(' >> ')[0])
cat_labels = list(data.set_index('image')['first_category'][descriptors_df.index])


km = KMeans(n_clusters=7).fit(descriptors_df)

comp = pd.DataFrame(index=set(cat_labels), columns=set(km.labels_)).fillna(0)
for i, img_id in enumerate(descriptors_df.index):
	comp.loc[cat_labels[i], km.labels_[i]] += 1

tsne_descriptors = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(descriptors_df)
tsne_descriptors.shape

import seaborn as sns
plt.figure(figsize=(12,12))
plt.title('Visualisation 2D t-SNE du tf-idf des descriptors')
sns.scatterplot(x=tsne_descriptors[:, 0], y=tsne_descriptors[:, 1], hue=cat_labels)
