import os
import yaml
import torch
import numpy as np
import cv2
import glob

from os import path
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------
# - Load Models -
# --> Load Propagation and Fusion 
from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
# --> Load Interaction
from RiTM.isegm.inference import utils as utils_RiTM

# ---------------------------------------------
# - Load DAVIS/ CiVOS/ myUtils/ Visualization -
# --> Load DAVIS Dataset
from dataset.davis_test_dataset import DAVISTestDataset
# --> Load DAVIS Interactive Session
from click_davisinteractive.session.session import DavisInteractiveSession
# --> Load CiVOS
from CiVOS_pipeline import DAVISProcessor
# --> Load myUtils
from my_utils.Click_generation_strategies import Click_Gen_Strat
# --> Visualization stuff
from my_utils.Image_Operations import PIL_DAVIS_image_operations_class
Display = PIL_DAVIS_image_operations_class()


# ------------------
# - Load Arguments -
parser = ArgumentParser()
parser.add_argument('--output')
parser.add_argument('--params', default="./evaluation_space/eval_debugging/DEBUGGING.yml")
args = parser.parse_args()


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
def prep_img_for_interaction_module(arg_davis):
    all_Image_path    = {}   # Find all sequences path's of DAVIS
    subset_sequences  = []   # Find only the relevant sequences to explore depending on the subset indicated
    subset_Image_path = {}   # Extract only the relevant sequences path's
    ALL_sequences     = (sorted(glob.glob(arg_davis + '/trainval/JPEGImages/480p/*')))

    for elem in ALL_sequences:
        name_sequence                 = elem.split('/')[-1]
        all_Image_path[name_sequence] = elem

    with open(arg_davis+ '/trainval/ImageSets/2017/val.txt', 'r') as lines:
        for line in lines:
            subset_sequences.append(line[:-1])

    for sequence in subset_sequences:
        subset_Image_path[sequence] = all_Image_path[sequence]

    #-
    return subset_Image_path


# ------------------------
# - Load Yaml Parameters -
print("YAML Parameters file used:",args.params.split("_")[-1].split(".")[0])
with open(args.params, 'rb') as f:
    yaml_params = yaml.load(f.read(), Loader=yaml.FullLoader)
# --> Visualization
Visu = yaml_params['Visualization'] # Display masks and other stuff to get a feeling of what is happening
if Visu:
    named_Win = "Propagation_output"
    cv2.namedWindow(named_Win, cv2.WINDOW_NORMAL)
# --> DAVIS Chall
max_interactions  = yaml_params['DAVIS_Chall']['max_interactions']
max_time          = yaml_params['DAVIS_Chall']['max_time']
subset            = yaml_params['DAVIS_Chall']['subset']
metric_2_optimize = yaml_params['DAVIS_Chall']['metric_to_optimize']
report_dir        = yaml_params['DAVIS_Chall']['report_davis_dir']
use_next_mask     = yaml_params['DAVIS_Chall']['use_next_mask']
# --> Standard params
arg_davis        = yaml_params['Standard_params']['path_2_davis']
arg_prop_model   = yaml_params['Standard_params']['prop_model']
arg_fusion_model = yaml_params['Standard_params']['fusion_model']
arg_inter_model   = yaml_params['Standard_params']['inter_model']
arg_save_mask    = yaml_params['Standard_params']['save_mask']
# --> Interaction params
inter_params = yaml_params['interaction_params']
# --> DAVIS params with Clicks
limit_for_points = yaml_params['limit_for_points']
CGS_strat        = yaml_params['CGS']    # Click generation strategies --> 1,2 and 3 as described in the paper
minimal_region_size       = yaml_params['minimal_region_size']
minimal_area_to_considere = yaml_params['minimal_area_to_considere']


# ---------
# - SETUP -
os.makedirs(args.output, exist_ok=True) # Create folder for output
torch.autograd.set_grad_enabled(False)  # Gradient calculation off


# -----------------------------------------
# - Click Generating Strategy Basic Setup -
CGS = Click_Gen_Strat()
CGS.set_strat(CGS_strat)
CGS.set_visualization(Visu, Display)


# ------------------------
# - Load DAVIS Sequences -
test_dataset= DAVISTestDataset(arg_davis+'/trainval', imset='2017/val.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
# --> Load all the images
images      = {}
num_objects = {}
for data in test_loader:
    rgb  = data['rgb']
    k    = len(data['info']['labels'][0])
    name = data['info']['name'][0]
    images[name]      = rgb
    num_objects[name] = k
print('Finished loading %d sequences.' % len(images))


# ------------------
# - Prepare Images -
subset_Image_path = prep_img_for_interaction_module(arg_davis)


# -----------------
# - Load Networks -
inter_model, prop_model, fusion_model = load_networks(arg_inter_model, arg_prop_model, arg_fusion_model)


# -----------------------------------------------
# - Set Some Variables for the DAVIS evaluation -
total_iter = 0
user_iter  = 0
last_seq   = None
pred_masks = None


def there_are_clicks(total_positive_clicks, total_negative_clicks):
    Obj_ids_exist = np.unique(np.append(total_positive_clicks, total_negative_clicks).reshape(-1, 3)[:, 0])
    return len(Obj_ids_exist) != 0

# -----------------------------
# - DAVIS Interactive Session -
with DavisInteractiveSession(davis_root          = arg_davis+'/trainval',
                             subset              = subset,
                             report_save_dir     = report_dir,
                             max_nb_interactions = max_interactions,
                             max_time            = max_interactions*max_time if max_time is not None else None,
                             metric_to_optimize  = metric_2_optimize) as sess:
    while sess.next():
        # ----------------------------------------
        # - Get GT Mask and Scribbles from DAVIS -
        sequence, scribbles, new_seq = sess.get_scribbles(only_last=True)
        gt_mask_davis_sequence, scribble_idx = sess.get_gt_masks()

        if new_seq:
            if Visu:
                Display.load_sequence_images(sequence)

            if 'processor' in locals():
                # Note that ALL pre-computed features are flushed in this step
                # We are not using pre-computed features for the same sequence with different user-id
                del processor # Should release some juicy mem

            frames_annotated = []
            print("\n---------------- New sequence : {0} ----------------\n".format(sequence))

            # ---------------------------------------------
            # - Setup CIVOS Params for Every New Sequence -
            processor = DAVISProcessor(prop_model, fusion_model, inter_model, images[sequence],
                                       num_objects[sequence])  # interaction -> propagation -> fusion
            processor.reset_interaction_count()
            processor.set_Visualization(Visu)
            processor.init_inter_mod_params(inter_params)
            processor.init_CGS(CGS_strat)

            # Save last time
            palette = Image.open(
                path.expanduser(arg_davis + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()
            if arg_save_mask:
                if pred_masks is not None:
                    seq_path = path.join(args.output, str(user_iter), last_seq)
                    os.makedirs(seq_path, exist_ok=True)
                    for i in range(len(pred_masks)):
                        img_E = Image.fromarray(pred_masks[i])
                        img_E.putpalette(palette)
                        img_E.save(os.path.join(seq_path, '{:05d}.png'.format(i)))

                if (last_seq is None) or (sequence != last_seq):
                    last_seq  = sequence
                    user_iter = 0
                else:
                    user_iter += 1

        # --------------------------------
        # - Find Current Annotated Frame -
        annotated_frame = None
        for idx, elem in enumerate(scribbles['scribbles']):
            if elem != []:
                annotated_frame = idx
        frames_annotated.append(annotated_frame)
        print("Annotated frame:", annotated_frame)


        # ------------------------------------------------
        # - Load Path to Images for the Current Sequence -
        Sequence_img_paths = sorted(glob.glob(subset_Image_path[sequence] + '/*'))  # Path to all images of the sequence
        path_2_img = Sequence_img_paths[annotated_frame]


        # -----------------------------------------
        # - Click Generating Strategy Advanced Setup -
        CGS.is_new_sequence(new_seq)
        CGS.set_annotate_frame(annotated_frame)
        CGS.get_GT_mask(gt_mask_davis_sequence)
        total_positive_clicks, total_negative_clicks = CGS.select_CGSy(scribbles, pred_masks, minimal_region_size,
                                                                       limit_for_points, minimal_area_to_considere)


        # -------------------
        # - Make Prediction -
        if there_are_clicks(total_positive_clicks, total_negative_clicks):
            # -> Set the clicks and frame
            processor.get_frame(path_2_img, annotated_frame)
            processor.reset_predictor_n_clicker()
            processor.set_positive_and_negative_clicks(total_positive_clicks, total_negative_clicks)
            # -> Interaction/Propagation/Fusion
            pred_masks, next_masks, this_idx = processor.CiVOS_interact()
        else:
            print('No positive nor negative clicks')
            pass    # Do nothing and just pass the mask computed from the previous round

        sess.submit_masks(pred_masks)


        # -----------------------------------------------------------
        # - Visualization # Display every frame after a propagation -
        if Visu:
            print("'--> Results of Propagation")
            processor.Visu_propagation(named_Win, pred_masks, alpha=0.3, v_time=0)

        total_iter += 1


    # --------------------------
    # - Get report and summary -
    report  = sess.get_report() #--> Most important element
    summary = sess.get_global_summary(save_file=path.join(args.output, 'summary.json'))
