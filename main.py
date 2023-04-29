import pandas as pd
import numpy as np
import os
import glob
import matplotlib.image as imread
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from PIL import Image


def checkImage(image, id,radius=2):
    #print(image)
    img = np.array(image)
    r = 0
    g = 1
    b = 2

    r_query = 250/255
    g_query = 170/255
    b_query = 31/255
    y, x = (np.where((img[:, :, r] >= r_query) & (img[:, :, g] >= g_query) & (img[:, :, b] <= b_query)))
    if len(y)>1:
        plt.imshow(image)
        plt.ylabel(id)
        plt.show()


if __name__ == '__main__':
    #read color that we want to search
    rightColor = imread.imread('right_color.jpg')[0][0]
    print(rightColor)

    #get our images that we want to scan
    data = os.listdir(glob.glob('./seed**/')[0])
    print(data)

    #create database that gets filled with names, which afterwards get associated with labels
    dataframe = pd.DataFrame(data, columns=['Name'])

    print(glob.glob('./seed**/')[0],data[0])
    print(data[0])
    #Image.open(glob.glob('./seed**/')[0]+data[0]).show()


    labels = []
    for image in data:
        print(image)
        labels.append(checkImage(imread.imread(glob.glob('./seed**/')[0]+image), image))





    # label_picture()
    # fill_label_in_database()
    # return database as csv