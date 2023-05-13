import pandas
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.image as imread
import csv
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from PIL import Image


def checkImage(image, id, min_pixel=3 ,radius=2):
    img = np.array(image)

    r_rightColor = rightColor[0]/255
    g_rightColor = rightColor[1]/255
    b_rightColor = rightColor[2]/255

    r_poleColor = poleColor[0]/255
    g_poleColor = poleColor[1]/255
    b_poleColor = poleColor[2]/255

    y, x = (np.where((img[:, :, 0] >= r_rightColor) & (img[:, :, 1] >= g_rightColor) & (img[:, :, 2] <= b_rightColor))) #pos of every right_color pixel
    if len(y) >= min_pixel: #checks, if we have at least min_pixel=3 yellow pixels
        #print(x,y)
        for i in range(len(y)):
            #print('Koord. y: '+str(max(0, y[i]-radius))+" "+str(min(y[i]+radius,(img.shape[0]-1))))
            #print('Koord. x: '+str(max(0, x[i]-1))+" "+str(min(x[i]+1,img.shape[1]-1)))
            slice = img[max(0, y[i]-radius):min(y[i]+radius, img.shape[0]-1)+1, max(0, x[i]-1):min(x[i]+1, img.shape[1]-1)+1]
            sl_y,sl_x = (np.where((slice[:, :, 0] >= r_rightColor) & (slice[:, :, 1] >= g_rightColor) & (slice[:, :, 2] <= b_rightColor)))
            if len(sl_y) >= min_pixel:
                #print("plot")
                #print(image)
                #plt.imshow(image)
                #plt.ylabel(id)
                #plt.show()


                #slice_pole = slice[max(0, y[i] - radius):min(y[i] + radius, slice.shape[0] - 1) + 1, x[i]:x[i]+1]
                #print(np.shape(slice))
                slice_pole = slice[:,1,:]
                #print(np.shape(slice_pole))

                slp_y = (np.where((slice_pole[:, 0] == r_poleColor) & (slice_pole[:, 1] == g_poleColor) & (slice_pole[:, 2] == b_poleColor)))
                #print(slp_y)
                #slp_y = np.squeeze(slp_y)
                #print(slp_y)
                if len(slp_y[0])>0:
                    #print('Koord. slp_y: ' + str(max(0, slp_y[i] - radius)) + " " + str(min(slp_y[i] + radius, (img.shape[0] - 1))))
                    #print('Koord. slp_x: ' + str(max(0, slp_x[i])) + " " + str(min(slp_x[i] + 1, img.shape[1])))
                    #print(slp_x)
                    print(id)
                    return 1

                #print(slice_pole)
                slf_y = (np.where((slice_pole[:, 0] >= r_rightColor) & (slice_pole[:, 1] >= g_rightColor) & (slice_pole[:, 2] <= b_rightColor)))

                #print(slf_y)
                if len(slf_y[0]) == ((radius*2)+1):
                    print(id)
                    return 1

    return 0


if __name__ == '__main__':
    #read color that we want to search
    rightColor = imread.imread('right_color.jpg')[0][0]
    poleColor = imread.imread('pole_color.jpg')[0][0]
    print("rightColor: " + str(rightColor))
    print("poleColor: " + str(poleColor))
    #get our images that we want to scan
    data = os.listdir(glob.glob('./seed**/')[0])
    print(data)

    print(glob.glob('./seed**/')[0],data[0])
    #Image.open(glob.glob('./seed**/')[0]+data[0]).show()

    imagefolder_path = glob.glob('./seed**/')[0]

    names = []
    labels = []
    for image in data:
        names.append((image[0:6]+'.jpg'))
        labels.append(checkImage(imread.imread(glob.glob('./seed**/')[0]+image), image))
    print(labels)

    dataframe = pandas.DataFrame()

    dataframe['Name']=names
    dataframe['Ampel']=labels
    dataframe.reset_index(drop=True)

    print(dataframe)
    dataframe.to_csv('dataframe.csv')



    # label_picture()
    # fill_label_in_database()
    # return database as csv