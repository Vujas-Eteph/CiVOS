import os
import yaml
import torch
import numpy as np
import cv2
import glob

from torchvision import transforms
from argparse import ArgumentParser
from PIL import Image

# ---------------
# - Load Models -
# --> Load Propagation and Fusion model
from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
# --> Load Interaction Model
from RiTM.isegm.inference import utils as utils_RiTM

# ---------------------------------------------
# - Load myUtils/ Visualization -
# --> Visualization stuff
from my_utils.Image_Operations import PIL_DAVIS_image_operations_class
Display = PIL_DAVIS_image_operations_class()


# ------------------
# - Load Arguments -
parser = ArgumentParser()
parser.add_argument('--params', default="Demo.yml")
args = parser.parse_args()

# --> Load CiVOS
from CiVOS_pipeline import DAVISProcessor


# --------------------------
# - Load Networks Function -
def load_networks(arg_inter_model, arg_prop_model, arg_fusion_model):
    # --> Load Interaction
    inter_model = utils_RiTM.load_is_model(arg_inter_model, f'cuda:0',
                                          cpu_dist_maps=False)  # False => Dist map is calculated without Cython
    # --> Load propagation
    prop_saved = torch.load(arg_prop_model)
    prop_model = PropagationNetwork().cuda().eval()
    prop_model.load_state_dict(prop_saved)
    # --> Load Fusion
    fusion_saved = torch.load(arg_fusion_model)
    fusion_model = FusionNet().cuda().eval()
    fusion_model.load_state_dict(fusion_saved)

    #-
    return inter_model, prop_model, fusion_model


# ------------------
# - Prepare Images -
def extract_imgs(path_2_video: str):
    img_files = sorted(glob.glob(path_2_video+'/*'))

    im_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im_transform = transforms.Compose([transforms.ToTensor(), im_normalization])

    images = []
    img_shape = np.array(Image.open(img_files[0]).convert('RGB')).shape
    images_tensor = torch.empty(1, len(img_files), img_shape[-1], img_shape[0], img_shape[1])
    for idx, img_file in enumerate(img_files):
        img = Image.open(img_file).convert('RGB')
        images.append(np.array(img))
        images_tensor[0,idx,:] = torch.unsqueeze(im_transform(img), dim = 0)

    images_arr = np.array(images)

    return img_files, images_arr, images_tensor


def Add_Mask_and_Image_Visu(images, predicted_masks, idx, alpha=0.5):
    mask = predicted_masks[idx].copy()
    mask_arr = np.zeros((mask.max() + 1, mask.shape[0], mask.shape[1]))
    for obj_id in range(0, mask.max() + 1):
        mask_arr[obj_id, :, :] = (mask == obj_id).astype(dtype=np.uint8)

    # Display the image
    img = images[idx].copy()

    img_zeros = np.zeros_like(img)

    # Draw mask on image
    colors = [[0,0,0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255],
              [255, 255, 255]]
    for m, color in zip(mask_arr, colors):
        for jdx, clr in enumerate(color):
            img_zeros[:, :, jdx] = img_zeros[:, :, jdx] + m * clr

    for jdx in range(0, img_zeros.shape[-1]):
        img[:, :, jdx] = img[:, :, jdx] * (1 - alpha) + img_zeros[:, :, jdx] * alpha

    return img


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('positif', x, y)
        positive_clicks_storage.append([obj_id, y,x])



        img_2_draw[idx_img] = cv2.circle(img_2_draw[idx_img], (x,y), radius=2, color=(0,250,0), thickness=-1)
        cv2.imshow(named_Win, img_2_draw[idx_img])


    if event == cv2.EVENT_RBUTTONDBLCLK:
        print('negatif', x, y)
        negative_clicks_storage.append([obj_id, y,x])

        img_2_draw[idx_img] = cv2.circle(img_2_draw[idx_img], (x,y), radius=2, color=(0,0,250), thickness=-1)
        cv2.imshow(named_Win, img_2_draw[idx_img])


# ------------------------
# - Load Yaml Parameters -
print("YAML Parameters file used:",args.params.split("_")[-1].split(".")[0])
with open(args.params, 'rb') as f:
    yaml_params = yaml.load(f.read(), Loader=yaml.FullLoader)

# --> Propagation params
arg_prop_model    = yaml_params['Standard_params']['prop_model']
arg_fusion_model  = yaml_params['Standard_params']['fusion_model']
arg_inter_model    = yaml_params['Standard_params']['inter_model']
# --> Interaction params
interaction_mod_params = yaml_params['interaction_params']


arg_video  = yaml_params['path_2_video']
nbr_of_obj = yaml_params['nbr_of_objects']


# ---------
# - SETUP -
torch.autograd.set_grad_enabled(False)  # Gradient calculation off


# ---------------------------
# - Load images for CiVOS -
img_files, imgs_arr, imgs_tsr = extract_imgs(arg_video)


# -----------------
# - Load Networks -
inter_model, prop_model, fusion_model = load_networks(arg_inter_model, arg_prop_model, arg_fusion_model)
processor = DAVISProcessor(prop_model, fusion_model, inter_model, imgs_tsr, nbr_of_obj)
processor.reset_interaction_count()
processor.init_inter_mod_params(interaction_mod_params)


named_Win = "Sequence"
cv2.namedWindow(named_Win, cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback(named_Win, click_and_crop)


positive_clicks_storage = []
negative_clicks_storage = []

idx_img = 0

obj_id = 1

img_2_draw = imgs_arr[:, :, :, ::-1].copy()

pred_masks = None

while True:
    print('Frame #', idx_img)
    if pred_masks is None:
        cv2.imshow(named_Win, img_2_draw[idx_img])
    else:
        img_w_mask = Add_Mask_and_Image_Visu(img_2_draw, pred_masks, idx_img)
        cv2.imshow(named_Win, img_w_mask)


    key = cv2.waitKey(0)

    if ord('y')==key:
        idx_img = idx_img - 1
        idx_img = idx_img%len(img_2_draw)

    if ord('x')==key:
        idx_img = idx_img + 1
        idx_img = idx_img%len(img_2_draw)

    if ord(' ')==key:
        if positive_clicks_storage == [] and negative_clicks_storage == []:
            print('No interactions')
        else:
            # Interaction mod prediction
            processor.get_frame(img_files[idx_img], idx_img)
            processor.set_img_resolution_manually([1,1])
            processor.reset_predictor_n_clicker()

            # Adapt clicks
            positive_clicks = np.array([[-1,-1,-1]])
            negative_clicks = np.array([[-1,-1,-1]])
            elem = None
            for elem in positive_clicks_storage:
                positive_clicks = np.append(positive_clicks, np.array([elem]), axis = 0)
            elem = None
            for elem in negative_clicks_storage:
                negative_clicks = np.append(negative_clicks, np.array([elem]), axis = 0)

            processor.set_positive_and_negative_clicks(positive_clicks[1:,:],
                                                       negative_clicks[1:,:])
            pred_masks, _ , _ = processor.CiVOS_interact()

        # empty clicks storage
        positive_clicks_storage = []
        negative_clicks_storage = []

    if ord('q')==key:
        cv2.destroyAllWindows()
        break

