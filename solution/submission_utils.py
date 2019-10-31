from pipeline.load_model import load_model
from pipeline.mask2lstr_v1 import mask2linestring
from pipeline.utils import process_img,process_path
from config import IMG_PATH,IMG_DIR,MODEL_PATH,IMG_SIZE,WRITE_TO_CSV
from glob import glob
from skimage.morphology import opening,closing,disk
from shapely.geometry import LineString
import os
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def submit_img(IMG_PATH,model,thresh,binary_thresh,start_channel,speeds):
  
  #speeds=[0,6.7056,8.3820,8.9408,10.0584,11.1760,13.4112,15.6464,20.1168,24.5872,29.0576]
  csv=[]
  speed=[]
  length=[]
  time=[]
  paths=[]
  if os.path.exists(IMG_PATH):
    #### retrieve metadata from IMG_PATH 
    code,city,chip=process_path(IMG_PATH)
    img=process_img(IMG_PATH)
    #img/=255.0
    #### prediction and thresholding block ####
    pred_mask=model.predict(np.expand_dims(img,axis=0))
    pred_mask[pred_mask>binary_thresh]=1.0
    pred_mask[pred_mask<binary_thresh]=0.0
    if np.sum(pred_mask[:,:,:,1:]) < 50:
      print('Non road image found')
      paths.append('AOI_{}_{}_chip{}'.format(code,city,chip))
      csv.append(LineString())
      speed.append(0)
      length.append(0)
      time.append(0)

      if WRITE_TO_CSV:
        df=pd.DataFrame({'ImageId':paths,'WKT_Pix':csv,'length_m':length,'travel_time_s':time})
        df.to_csv('csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh),index=False)
      if os.path.exists('csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh)):
        print('csv file saved to csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh))
      return None

    pred_mask=np.argmax(pred_mask[0,:,:,:],axis=2)
    cv2.imwrite('cache/temp.png',pred_mask*20)
    cv2.imwrite('cache/temp_closes.png',pred_mask*20)
    plt.imshow(opening(pred_mask))
    plt.show()
    plt.imshow(pred_mask)
    plt.show()
    plt.imshow(closing(pred_mask))
    plt.show()
    
    #plt.savefig('temp.png')
    for i in range(start_channel,11):
        curr_mask=np.zeros(shape=(512,512))
        curr_mask[pred_mask==i]=1.0
        curr_mask=closing(curr_mask,disk(4))
        #### resize 512,512 mask to 1300,1300
        curr_mask=cv2.resize(curr_mask,(1300,1300))
        if np.sum(curr_mask)>1:
            cv2.imwrite('cache/temp_{}.png'.format(i),curr_mask*255.0)
            curr_mask=cv2.imread('cache/temp_{}.png'.format(i),0).astype(np.float32)
            curr_mask[curr_mask>0]=1.0
        #curr_mask=curr_mask.astype(np.float32)
        #plt.title(str(i))
        #plt.imshow(curr_mask,cmap='gray')
        #plt.show()
        #### mask to linestring conversion ####
        
        data=mask2linestring(curr_mask.T,thresh=thresh,plot=False,write2csv=False)
        for data_ in data:
                if data_.length > 5:
                    paths.append('AOI_{}_{}_chip{}'.format(code,city,chip))
                    csv.append(data_)
                    speed.append(speeds[i])
                    #### convert euclidean distance to 400 metres [400/1300]
                    length.append(data_.length*0.31)
                    time.append(float(data_.length*0.31)/float(speeds[i]))
    
    if WRITE_TO_CSV:
        df=pd.DataFrame({'ImageId':paths,'WKT_Pix':csv,'length_m':length,'travel_time_s':time})
        df.to_csv('csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh),index=False)
    if os.path.exists('csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh)):
        print('csv file saved to csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh))

def submit_dir(DIR_PATH,model,thresh,binary_thresh,start_channel,speeds):
  #### predicting from folder #####
  
  #speeds=[0,6.7056,8.3820,8.9408,10.0584,11.1760,13.4112,15.6464,20.1168,24.5872,29.0576]
  if os.path.exists(DIR_PATH):
      print(True)
  else:
    print('check dir path')
    return None

  for IMG_PATH in tqdm(sorted(glob(DIR_PATH+'/*'))):
      csv=[]
      speed=[]
      length=[]
      time=[]
      paths=[]
      csv_paths=[]
      print(IMG_PATH)
      code,city,chip=process_path(IMG_PATH)
      img=process_img(IMG_PATH)
      #### prediction and thresholding block ####
      pred_mask=model.predict(np.expand_dims(img,axis=0))
      pred_mask[pred_mask>binary_thresh]=1.0
      pred_mask[pred_mask<binary_thresh]=0.0
      if np.sum(pred_mask[:,:,:,1:]) < 50:
        print('Non road image found')
        if 'San_Juan' in city:
          paths.append('SN5_roads_test_AOI_{}_{}_chip{}'.format(code,city,chip))
        else:
          paths.append('SN5_roads_test_public_AOI_{}_{}_chip{}'.format(code,city,chip))
        csv.append('LINESTRING EMPTY')
        speed.append(0)
        length.append(0)
        time.append(0)
        pred_mask=np.argmax(pred_mask[0,:,:,:],axis=2)
        cv2.imwrite('cache/temp.png',pred_mask*20)
        cv2.imwrite('cache/temp_closes.png',pred_mask*20)
        plt.title('Empty Road')
        plt.imshow(pred_mask)
        plt.show()
        if WRITE_TO_CSV:
          df=pd.DataFrame({'ImageId':paths,'WKT_Pix':csv,'length_m':length,'travel_time_s':time})
          df.to_csv('csv/dir_csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh),index=False)
        if os.path.exists('csv/dir_csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh)):
          print('csv file saved to dir_csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh))
        
      else:
        pred_mask=np.argmax(pred_mask[0,:,:,:],axis=2)
        cv2.imwrite('cache/temp.png',pred_mask*20)
        cv2.imwrite('cache/temp_closes.png',pred_mask*20)
        plt.title('Positive Road')
        plt.imshow(pred_mask)
        plt.show()

        #plt.savefig('temp.png')
        for i in range(start_channel,11):
          curr_mask=np.zeros(shape=(512,512))
          curr_mask[pred_mask==i]=1.0
          curr_mask=closing(curr_mask,disk(4))
          #### resize 512,512 mask to 1300,1300
          curr_mask=cv2.resize(curr_mask,(1300,1300))
          if np.sum(curr_mask)>1:
              cv2.imwrite('cache/temp_{}.png'.format(i),curr_mask*255.0)
              curr_mask=cv2.imread('cache/temp_{}.png'.format(i),0).astype(np.float32)
              curr_mask[curr_mask>0]=1.0
          #curr_mask=curr_mask.astype(np.float32)
          #plt.title(str(i))
          #plt.imshow(curr_mask,cmap='gray')
          #plt.show()
          #### mask to linestring conversion ####
        
          data=mask2linestring(curr_mask.T,thresh=thresh,plot=False,write2csv=False)
          for data_ in data:
                  if data_.length > 2:
                      if 'San_Juan' in city:
                        paths.append('SN5_roads_test_AOI_{}_{}_chip{}'.format(code,city,chip))
                      else:
                        paths.append('SN5_roads_test_public_AOI_{}_{}_chip{}'.format(code,city,chip))
                        #paths.append('AOI_{}_{}_chip{}'.format(code,city,chip))
                      csv.append(data_)
                      speed.append(speeds[i])
                      #### convert euclidean distance to 400 metres [400/1300]
                      length.append(data_.length*0.307)
                      time.append(float(data_.length*0.307)/float(speeds[i]))
    
        if WRITE_TO_CSV:
          df=pd.DataFrame({'ImageId':paths,'WKT_Pix':csv,'length_m':length,'travel_time_s':time})
          df.to_csv('csv/dir_csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh),index=False)
          csv_paths.append('csv/dir_csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh))
        if os.path.exists('csv/dir_csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh)):
          print('csv file saved to dir_csv/{}_{}_{}_{}.csv'.format(code,city,chip,thresh))
  return csv_paths

def combine_submissions(CSV_DIR):
  '''
  ref : stackoverflow 
  '''
  #CSV_DIR='csv/dir_csv/*'
  filenames=sorted(glob(CSV_DIR+'/*.csv'))
  print(filenames)
  combined_csv = pd.concat( [ pd.read_csv(f) for f in filenames ] )
  combined_csv.to_csv('submission.csv',index=False)
