import matplotlib.pyplot as plt 
import os
import sys

img_path = "nlpchatbot/htg_data/test/0ad9e7dfb.png"

af = plt.imread(img_path)
print(af.shape)