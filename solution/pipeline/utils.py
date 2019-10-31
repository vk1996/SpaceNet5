from keras.preprocessing.image import load_img,img_to_array
from config import CITIES,CODES,IMG_SIZE

def process_img(path):
    img=load_img(path,target_size=(IMG_SIZE,IMG_SIZE))
    img=img_to_array(img)
    return img/255.0

def process_path(path):
    for city in CITIES:
        if city in path:
            code=CODES[CITIES.index(city)]
            chip=path.split('chip')[-1][:-4]
            return (code,city,chip)
            
    
