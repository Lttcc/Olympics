import matplotlib.pyplot as plt
import imageio,os
from PIL import Image
GIF=[]
filepath=os.getcwd()+"\\img"
filenames=os.listdir(filepath)
filenames.sort(key= lambda x:int(x[:-4]))
for filename in filenames:
    GIF.append(imageio.imread(filepath+"\\"+filename))
imageio.mimsave(os.getcwd()+"\\"+'result.gif',GIF,duration=0.05)#这个duration是播放速度，数值越小，速度越快
