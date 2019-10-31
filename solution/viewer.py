import pandas as pd
from shapely.geometry import LineString
from matplotlib import pyplot as plt
import numpy as np
import cv2
from keras.preprocessing.image import load_img,img_to_array
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("csv")
parser.add_argument("img")
args=parser.parse_args()
path=args.csv
img_path=args.img

#'input/SN5_roads_train_AOI_7_Moscow_PS-RGB_chip134.tif'
#'csv/7_Moscow_134_10.csv'
#path=input('Enter csv path:')
#img_path=input('Enter img path:')
img=load_img(img_path,target_size=(1300,1300))
img=img_to_array(img)
img=img.astype(np.uint8)
#fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(5,5),sharey=True)

df=pd.read_csv(path)
print(df.head())
chars=')LINESTRING(,  '
speed_dict={'6':(0,0,100),'8':(0,100,0),'10':(100,0,0),'11':(0,100,100),'13':(100,0,100),'15':(100,100,0),'20':(0,100,200),'24':(0,200,100),'29':(200,0,100)}

def str2lstr(str_):
    ### strip all meta string ####
    points=str_.strip(chars)
    points=points.strip('(')
    ### retrieve and stack points by splitting ',' ###
    points=points.split(',')
    points_=[]
    for i in points:
        if len(i.split(' ')) == 3:
            _,x,y=i.split(' ')
        else:
            x,y=i.split(' ')
        
        points_.append((float(x),float(y)))

    return LineString(points_)

#ax.imshow(img.astype(np.float16))
for i in range(len(df)):
    line=str2lstr(df['WKT_Pix'][i])
    speed=int(df['length_m'][i]/df['travel_time_s'][i])
    color=speed_dict[str(speed)]
    x,y=line.xy
    count=0
    for point_x,point_y in zip(x,y):
        curr_=(int(point_x),int(point_y))
        
        if count>0:
            cv2.line(img,prev_,curr_,color,3)
            prev_=curr_
        else:
            cv2.circle(img,curr_,5,(255,255,255),3)
            prev_=curr_
            count+=1

plt.imsave('temp.png',img)  
#plt.show()  
