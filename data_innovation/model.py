from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data_gen import create_texts


# Getting data 
word_vectors = create_texts(10000)

# Defining Model
model = TSNE(learning_rate=100)

# Fitting Model
transformed = model.fit_transform(word_vectors)

# Plotting 2d t-Sne
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

plt.scatter(x_axis, y_axis)
plt.show()