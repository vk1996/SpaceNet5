import cv2
import numpy as np
from matplotlib import pyplot as plt


def create_seq_crop(img,crop_size):
    '''
    input: rgb image
    output: 9,448,448,3
    '''
    img=cv2.resize(img,(1344,1344))
    height,width=img.shape[-3],img.shape[-2]
    dy,dx=crop_size[0],crop_size[1]
    heights=[0,448,896]
    widths=heights
    #create a zero array of n crops of req shape
    cropped_array=np.zeros(shape=(int(height/dy)**2,dy,dx,img.shape[-1]))
    i=0
    for y in heights:
        for x in widths:
            cropped_array[i]=img[y:(y+dy), x:(x+dx), :]
            i+=1
    
    return cropped_array

def merge_seq_crop(cropped,out_shape):
    '''
    input: 9,448,448,3
    output: rgb 
    '''
    #create a single image which will store n crops
    merge=np.zeros((out_shape[0],out_shape[1],out_shape[2]))
    height,width=cropped.shape[-3],cropped.shape[-2]
    dy,dx=cropped.shape[-2],cropped.shape[-3]
    heights=[0,448,896]
    widths=heights
    i=0
    for y in heights:
        for x in widths:
            merge[y:(y+dy), x:(x+dx), :]=cropped[i,:,:,:]
            i+=1
    return merge    
'''   
im=cv2.imread('SN5_roads_train_AOI_7_Moscow_PS-RGB_chip134.tif')
cropped=create_seq_crop(im,(448,448))

fig,ax=plt.subplots(nrows=1,ncols=9,figsize=(30,30),sharey=True)
i=0
for crop in cropped:
    if len(crop.shape)==4:
        ax[i].imshow(np.squeeze(crop).astype(np.int32))
    else:
        ax[i].imshow(crop.astype(np.int32))
    i+=1
#plt.show()
merge=merge_seq_crop(cropped,(1344,1344,3))
cv2.imwrite('merged.png',merge.astype(np.uint8))
'''        
    
