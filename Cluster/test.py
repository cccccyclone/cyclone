# -*- coding: utf-8 -*-
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
image = io.imread('panda.jpg')
io.imshow(image)
io.show()
rows = image.shape[0]
cols = image.shape[1]
image = image.reshape(image.shape[0]*image.shape[1],3)
print len(image)
kmeans = KMeans(n_clusters = 16, n_init=10, max_iter=10)
kmeans.fit(image)
clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
labels = np.asarray(kmeans.labels_,dtype=np.uint8 )
print len(labels)
labels = labels.reshape(rows,cols);

image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        image[i, j, :] = clusters[labels[i, j], :]

io.imsave('reconstructed_panda.jpg', image);
io.imshow(image)
io.show()