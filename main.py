from PIL import Image
import pandas as pd
import numpy as np
import os
import glob

def checkRange(image: Image, length, min): #checks, if right


def checkImage(image: Image):
    for x in range(0,image.width):
        for y in range(0,image.height):
            if image.getpixel((x,y))==rightColor:
                print('a')


if __name__ == '__main__':
    #read color that we want to search
    rightColor = Image.open('right_color.jpg').getpixel((0,0))
    print(rightColor)

    #get our images that we want to scan
    data = os.listdir(glob.glob('./seed**/')[0])

    #create database that gets filled with names, which afterwards get associated with labels
    dataframe = pd.DataFrame(data, columns=['Name'])

    labels = []
    for image in data:
        labels.append(checkImage(image))




    # label_picture()
    # fill_label_in_database()
    # return database as csv