from keras import models
from segmentation_models.losses import DiceLoss,CategoricalFocalLoss
#from classification_models.common.blocks import Slice

def load_model(path):
    dice_loss = DiceLoss()
    per_img_dice_loss=DiceLoss(per_image=True)
    wt_dice_loss=DiceLoss(class_weights=[1,2,2,2,1,2,1,2,1,1,1])
    cat_loss = CategoricalFocalLoss()
    #bce=binary_crossentropy
    total_loss = dice_loss + (1 *cat_loss)
    return models.load_model(path,custom_objects={'dice_loss_plus_1focal_loss':total_loss,'dice_loss':DiceLoss()})
