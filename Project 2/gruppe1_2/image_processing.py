# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:49:42 2017

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



class image_processor:
    def __init__(self, file_folder="",file_extension=".bmp"):
        self.folder = file_folder
        self.extension = file_extension


    def read_image(self,name,fullFOV=True):
        im = Image.open(self.folder + name + self.extension)

        if fullFOV:
            return np.copy(np.array(im))
        elif not fullFOV:
            return self.get_center(np.array(im))

    def read_images(self,names,fullFOV=True):
        ims = []
        for name in names:
            ims.append(self.read_image(name,fullFOV))

        return np.array(ims)

    def make_histogram(self,image, title=""):
        n, bins, patches = plt.hist(image.flatten(),normed=True)
        plt.title(title)
        plt.xlabel("Pixel Value")
        plt.ylabel("Distribution")
        plt.show()

    def get_extrema(self, image):
        return np.min(image), np.max(image)

    def get_position_extrema(self,image):

        return np.unravel_index(np.argmin(image),image.shape), \
                    np.unravel_index(np.argmax(image),image.shape)

    def get_mean(self,image):
        return np.mean(image)

    def average_image(self,images):
        return np.mean(images,axis=0)
        im_sum = images[0]
        for i in images[1:]:
            im_sum += i

        return im_sum/images.shape[0]


    def get_std(self,image): #get std...
        return np.std(image)

    def get_center(self, image, size=300):
        image_shape = image.shape
        center = (int(image_shape[0]/2),int(image_shape[1]/2))
        return np.copy(image[center[0] - int(size/2):center[0] + int(size/2),\
                         center[1] - int(size/2):center[1] + int(size/2)])


    def process_image_mean_and_noise(self,image_names):
        b1 = self.read_image(image_names[0])
        b2 = self.read_image(image_names[1])

        b_sum = b1 + b2
        b_center = self.get_center(b_sum)
        b_center_mean = self.get_mean(b_center)

        b_sub = b1-b2
        b_center_sub = self.get_center(b_sub)
        b_center_std = self.get_std(b_center_sub)

        return b_center_mean, b_center_std

    def get_g(self,bias_names,flat_names):
        f_mean,f_std = self.process_image_mean_and_noise(flat_names)
        b_mean,b_std = self.process_image_mean_and_noise(bias_names)

        return (f_mean - b_mean)/(f_std**2 - b_std**2)
    
    def get_RON(self,bias_names,flat_names):
        bias_noise = np.std(self.read_image(bias_names[0]))
        g = self.get_g(bias_names,flat_names)
        print(bias_noise)
        
        return g*bias_noise
        

    def get_noise(self,image_names, save=False):
        even_sum = self.read_image(image_names[0])
        odd_sum = self.read_image(image_names[1])

        if save:
            noises= np.zeros(int(len(image_names)/2))
            noises[0] = self.get_std(odd_sum-even_sum)

        for i in range(2,len(image_names),2):
            even_sum += self.read_image(image_names[i])
            odd_sum += self.read_image(image_names[i+1])
            if save:
               noises[int(i/2)] = self.get_std(odd_sum-even_sum) /(i/2)

        if save:
            return noises, self.get_std(odd_sum-even_sum)

        return self.get_std(odd_sum-even_sum)

    def get_picture_slice(self,name):
        if isinstance(name,np.ndarray):
            data = name
        else:
            data = self.read_image(name,fullFOV=True)

        return data[int(data.shape[0]/2),:]

    def plot_picture_slice(self,name):
        data = self.get_picture_slice(name)
        
        plt.plot(data)
        plt.title("Slice of the Center of the Cleaned Diffraction pattern")
        plt.xlabel("Pixel")
        plt.ylabel("Pixel value")
        plt.show()




    def clean_image(self,I_raw_name,raw_dark_names,flat_names,flat_dark_names,fullFOV=True):
        I_raw = self.read_image(I_raw_name,fullFOV=fullFOV)
        raw_darks = self.read_images(raw_dark_names,fullFOV=fullFOV)
        raw_dark_avr = self.average_image(raw_darks)

        flats = self.read_images(flat_names,fullFOV=fullFOV)


        flat_darks = self.read_images(flat_dark_names,fullFOV=fullFOV)

        flat_dark_avr = self.average_image(flat_darks)
        flat_avr = self.average_image(flats)

        flat_master = flat_avr - flat_dark_avr


        flat_master_norm = flat_master/self.get_mean(flat_master)

        return (I_raw - raw_dark_avr)/(flat_master_norm)


    def save_image(self,image,name):
        im = Image.fromarray(np.uint8(np.where(image>255, 255,image)))
        im.save(name)


    def plot_color_hist(self,name):
        data = self.read_image(name)


        value_count = np.zeros((3,256))
        for i in range(3):
            for val in range(256):
                value_count[i,val] = np.sum(data[:,:,i] == val)

        print("The maximum value for red is: ",np.max(data[:,:,0]))
        print("The maximum value for blue is: ",np.max(data[:,:,2]))
        print("The maximum value for green is: ",np.max(data[:,:,1]))

        plt.plot(value_count[0],"r")
        plt.plot(value_count[1],"g")
        plt.plot(value_count[2],"b")

        plt.show()

    def plot_highest_row_color(self,name,title=""):
        data = self.read_image(name)

        print("The maximum value for red is: ",np.max(data[:,:,0]))
        print("The maximum value for blue is: ",np.max(data[:,:,2]))
        print("The maximum value for green is: ",np.max(data[:,:,1]))

        data = np.mean(data,axis=0)

        plt.plot(data[:,0],"r",label="Red")
        plt.plot(data[:,1],"g",label="Green")
        plt.plot(data[:,2],"b",label="Blue")
        plt.title(title)
        plt.xlabel("Pixel")
        plt.ylabel("Average pixel count over row")
        plt.legend()

        plt.show()
        
    def convert_images(self,names):
        data = self.read_images(names)
        
        for i,im in enumerate(data):
            self.save_image(im,names[i] + ".png")



if __name__=="__main__":
    ip = image_processor()
    bias = ip.read_image("bf1")
    dark = ip.read_image("df_max_exp")

    ip.make_histogram(bias,title="Histogram for the Bias")
    ip.make_histogram(dark,title="Histogram for the Dark Frame")

    print("For bias: min = {}, max = {}".format(ip.get_extrema(bias)[0],ip.get_extrema(bias)[1]))
    print("For dark frame: min = {}, max = {}".format(ip.get_extrema(dark)[0],ip.get_extrema(dark)[1]))

    print("For bias, the position is: min = {}, max = {}".format(ip.get_position_extrema(bias)[0],ip.get_position_extrema(bias)[1]))
    print("For dark frame, the position is: min = {}, max = {}".format(ip.get_position_extrema(dark)[0],ip.get_position_extrema(dark)[1]))

    print("For bias: mean = {}".format(ip.get_mean(bias)))
    print("For dark frame: mean = {}".format(ip.get_mean(dark)))
    
    print("Statistics for bias and flat")
    print("----------------------------")
    
    bias_mean, bias_std = ip.process_image_mean_and_noise(["bf1","bf2"])
    flat_mean, flat_std = ip.process_image_mean_and_noise(["ff2","ff4"])
    
    print("For bias: mean = {}, std = {}".format(bias_mean,bias_std))
    print("For flat: mean = {}, std = {}".format(flat_mean,flat_std))
    


    


    print("The convertion constant: g = ",ip.get_g(["bf1","bf2"],["ff2","ff4"]))
    print("The readout noise RON = ",ip.get_RON(["bf1","bf2"],["ff2","ff4"]))
    
    print("Plotting noise over flates")
    print("--------------------------")
    
    flat_frames = ["ff" + str(i) for i  in range(1,17)]
    ns = np.arange(1,9)
    noise = ip.get_noise(flat_frames,save=True)[0]
    expected_noise = noise[0]*1/np.sqrt(ns)
    plt.plot(ns,noise,label="Actual")
    plt.plot(ns,expected_noise,label="Expected")
    plt.title("Noise of the Flat Frames")
    plt.xlabel("Number of Pairs n")
    plt.ylabel(r"Normalized Noise $\sigma_{(F_1 + \ldots F_{2n})- (F_2 + \ldots F_{2n+1})}/n$")
    plt.show()
    
    print("Cleaning")
    print("--------")
    
    
    raw = "3_2"
    raw_darks = ["df" + str(i) + "_4" for i in range(1,6)]
    flat_frames = ["ff" + str(i) for i  in range(1,17)]
    flat_darks = ["df_ff" + str(i) for i in range(1,6)]
    corr_I = ip.clean_image(raw,raw_dark_names=raw_darks,flat_names=flat_frames,flat_dark_names=flat_darks,fullFOV=True)
    #ip.save_image(corr_I,"corrected_image_fov.png")
    
    
    print("Slicing and Color Images")
    print("------------------------")

    ip.plot_picture_slice("3_1")

    #ip.plot_color_hist("rød fokus")
    print("For Green Focus:")
    ip.plot_highest_row_color("grønt fokus",title="Distribution for Green Light")
    print("For Red Focus:")
    ip.plot_highest_row_color("rød fokus",title="Distribution for Red Light")
    print("For Blue Focus:")
    ip.plot_highest_row_color("blått",title="Distribution for Blue Light")
    
    ip.convert_images(["df_max_exp"])

    """
    Vi lagde flat field ved å sette ark foran kameraet,
    men det kan være støy på linsen, hvilket da ikke kommer med
    på flat fielden!!!!!!!!
    """