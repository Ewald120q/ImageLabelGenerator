import pandas
import numpy as np
import os
import glob
import matplotlib.image as imread
import sys


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
            print(f"x:{x}")

            if x[i] == 0 or x[i] == img.shape[1]-1:
                print("dennis")
                stripe = img[:,x[i],:]
                print(stripe.shape)
                stripe_y = (np.where((stripe[ :, 0] >= r_rightColor) & (stripe[ :, 1] >= g_rightColor) & (stripe[ :, 2] <= b_rightColor)))
                for j in range(radius, img.shape[0]-(radius+1)):
                    stripe_slice = stripe[j-radius: j+radius+1,:]
                    stsly = np.where((stripe_slice[:, 0] >= r_rightColor) & (stripe_slice[:, 1] >= g_rightColor) & (
                                stripe_slice[:, 2] <= b_rightColor))
                    print(stsly)
                    if len(stsly[0]) == ((radius*2)+1):
                        return 1



            else:
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

                        if slice_pole.shape[1]==2:
                            slice
                        #print(np.shape(slice_pole))

                        slp_y = (np.where((slice_pole[:, 0] == r_poleColor) & (slice_pole[:, 1] == g_poleColor) & (slice_pole[:, 2] == b_poleColor)))
                        #print(slp_y)
                        #slp_y = np.squeeze(slp_y)
                        #print(slp_y)
                        if len(slp_y[0])>0:
                            #print('Koord. slp_y: ' + str(max(0, slp_y[i] - radius)) + " " + str(min(slp_y[i] + radius, (img.shape[0] - 1))))
                            #print('Koord. slp_x: ' + str(max(0, slp_x[i])) + " " + str(min(slp_x[i] + 1, img.shape[1])))
                            #print(slp_x)
                            #print(id)
                            return 1

                        #print(slice_pole)
                        slf_y = (np.where((slice_pole[:, 0] >= r_rightColor) & (slice_pole[:, 1] >= g_rightColor) & (slice_pole[:, 2] <= b_rightColor)))

                        #print(slf_y)
                        if len(slf_y[0]) == ((radius*2)+1):
                            #print(id)
                            print(slice)
                            #left_from_poleslice = slice[:, 0, :]
                            #right_from_poleslice = slice[:, min(2, slice.shape[0] - 1), :]

                            outside_poleslice = np.delete(slice, 1, axis=0)

                            lory = len(sl_y) - ((radius*2)+1)

                            if lory > 0:
                                return 1

    return 0


if __name__ == '__main__':
    #read color that we want to search
    rightColor = imread.imread('right_color.jpg')[0][0]
    poleColor = imread.imread('pole_color.jpg')[0][0]
    print("rightColor: " + str(rightColor))
    print("poleColor: " + str(poleColor))
    #get our images that we want to scan
    if len(sys.argv) == 1:  # when no arugment passed
        data = os.listdir(glob.glob('./seed**/')[0])
    else:
        data = os.listdir(sys.argv[1])
    print(data)

    print(glob.glob('./seed**/')[0],data[0])
    #Image.open(glob.glob('./seed**/')[0]+data[0]).show()

    imagefolder_path = ""

    if len(sys.argv) == 1: #when no arugment passed
        imagefolder_path = glob.glob('./seed**/')[0]
    else:
        imagefolder_path = sys.argv[1] #when argument passed, we take it as foldername

    names = []
    labels = []
    for image in data:
        names.append((image[0:6]+'.jpg'))
        print(image[0:6]+'.jpg')
        if len(sys.argv) == 1:
            labels.append(checkImage(imread.imread(glob.glob('./seed**/')[0]+image), image))
        else:
            labels.append(checkImage(imread.imread(sys.argv[1] + image), image))
    print(labels)

    dataframe = pandas.DataFrame()

    dataframe['Name']=names
    dataframe['Ampel']=labels
    dataframe.reset_index(drop=True)

    print(dataframe)
    print(str(imagefolder_path)[2:-1])
    dataframe.to_csv(f'dataframe_{(str(imagefolder_path))[2:-1]}.csv') #Array cuts Slashes, so that it does not get interpreted as path



    # label_picture()
    # fill_label_in_database()
    #return database as csv