import torch
import numpy as np

from inference_core import InferenceCore

from util.tensor_util import pad_divide_by, compute_tensor_iou
from model.aggregate import aggregate_wbg, aggregate_wbg_vujas

import cv2
from RiTM.isegm.inference.predictors import get_predictor
from RiTM.isegm.inference.clicker import Clicker, Click
from PIL import Image
from torchvision import transforms


class DAVISProcessor:
    """
    Acts as the junction between DAVIS interactive track and our inference_core
    """
    def __init__(self, prop_net, fuse_net, inter_net, images, num_objects, device='cuda:0'):
        self.inter_net = inter_net.to(device, non_blocking=True)

        # self.out_masks = np.array([0])
        self.propagated_masks = np.array([0])

        images, self.pad = pad_divide_by(images, 16, images.shape[-2:])
        self.device = device

        # Padded dimensions
        nh, nw = images.shape[-2:]
        self.nh, self.nw = nh, nw


        self.get_number_of_obj_instances(num_objects)

        self.propagation = InferenceCore(prop_net, fuse_net, images, num_objects, mem_profile=0, device=device)

        self.Visualization_switch = False   # Set default visualization to default


    def reset_interaction_count(self):
        self.interacted_count = 0


    def get_number_of_obj_instances(self, nbr_of_obj_in_sequence):
        '''
            nbr_of_obj_in_sequence: int
        '''
        self.total_nbr_of_obj_in_sequence = nbr_of_obj_in_sequence
        self.indices_of_obj_in_sequence   = np.arange(self.total_nbr_of_obj_in_sequence)+1 # vector : [1,2,...,n object]


    def set_Visualization(self, Visualization_switch):
        '''
        Visualization_switch: True/False, use visualization functions
        '''
        self.Visualization_switch = Visualization_switch


    def init_inter_mod_params(self, inter_mod_params):
        self.brs_mode        = inter_mod_params['brs_mode']
        self.use_soft_switch = inter_mod_params['use_soft_mask']
        self.pred_thr        = inter_mod_params['pred_thr']
        self.use_previous_mask_switch     = inter_mod_params['prev_mask']
        self.use_soft_in_prev_mask_switch = inter_mod_params['use_soft_in_prev_mask']

        if not self.use_previous_mask_switch:           # If self.use_previous_mask_switch false, then always have keep_transf_switch to false
            self.use_soft_in_prev_mask_switch = False   # If previous mask is not used, then no need to use the soft_in_prev_mask_switch

        self.agg_vujas = inter_mod_params['aggregation_vujas']

    def init_CGS(self, CGS_strat):
        self.CGS_strat = CGS_strat


    def get_frame(self, path_2_img: str, annotated_frame_idx: int):
        '''
        -> Get the annotated frame idx.
            path_2_sequence    : path to the image
            annotated_frame_idx: int, number of the frame annotated
        '''
        self.anno_f_idx     = annotated_frame_idx
        self.Ori_img        = np.array(Image.open(path_2_img))
        self.Ori_img_tensor = transforms.ToTensor()(self.Ori_img.copy()).to(self.device)
        self.img_shape      = self.Ori_img.shape
        self.img_res        = self.Ori_img.shape[:2]

        # Create an empty mask for the interaction mask output of shape[#obj, 1, H, W]
        self.mask_interaction_output = torch.zeros((self.total_nbr_of_obj_in_sequence, 1, self.img_shape[0], self.img_shape[1]),
                                            dtype=torch.float32, device=self.device)

    def set_img_resolution_manually(self, new_img_res: list):
        '''Only used in Demo'''
        self.img_res = np.array([new_img_res])


    def reset_predictor_n_clicker(self):
        self.interaction_predictor = get_predictor(self.inter_net, self.brs_mode, self.device, prob_thresh=self.pred_thr)
        self.interaction_predictor.set_input_image(self.Ori_img)
        self.inter_mod_clicker = Clicker()


    def set_previous_mask_in_interaction_module(self, pseudo_prev_mask):
        '''
        -> Set previous predicted mask for interaction on the current frame and object
            pseudo_prev_mask: np.array, shape: (1,1,H,W), dtype: float.32
        '''
        self.interaction_predictor.prev_prediction = torch.tensor(pseudo_prev_mask, device='cuda:0', dtype=torch.float32)


    def get_interaction_prediction(self):
        '''
        -> Get soft mask prediction from the interaction module. Soft mask is the raw input from the interaction module, where the mask
        is a probability map.
            hard_mask = soft_mask > threshold
        '''
        soft_mask  = self.interaction_predictor.get_prediction(self.inter_mod_clicker)
        hard_mask  = soft_mask > self.pred_thr
        interaction_module_state = self.interaction_predictor._get_transform_states()
        #-
        return soft_mask, hard_mask, interaction_module_state


    def interaction_predictionn(self, Obj_id):
        '''
        -> Add the interaction module prediction mask of each object in the place holder mask_interaction_output.
            Obj_id: int
        '''
        soft_mask, hard_mask, _ = self.get_interaction_prediction()
        interaction_mask_to_use        = soft_mask if self.use_soft_switch else hard_mask

        # Put prediction for current object ot the global interaction mask that englobes all present instances
        self.mask_interaction_output[Obj_id - 1] = torch.tensor(interaction_mask_to_use,
                                                         # Add the predicted mask for our output for the propagation module
                                                         dtype=torch.float32,
                                                         device=self.device).unsqueeze(dim=0)


    def set_positive_and_negative_clicks(self, total_positive_clicks, total_negative_clicks):
        '''
        setup positive and negative clicks
        :return:
        '''
        self.get_positive_interactions(total_positive_clicks)
        self.get_negative_interactions(total_negative_clicks)


    def get_positive_interactions(self, positive_interactions):
        """
        --> Get positive interactions:
            positive_interactions: np.array([int(obj_id#1), H, W], [int(obj_id#2), H, W],...)
        <- Return
            positive_clicks_for_interaction: np.array([int(obj_id#1), H, W], [int(obj_id#2), H, W],...)
        """
        positive_interactions[:,1:] *= self.img_res
        self.positive_clicks_for_interaction = positive_interactions


    def get_negative_interactions(self, negative_interactions):
        """
        --> Get positive interactions:
            positive_interactions: np.array([int(obj_id#1), H, W], [int(obj_id#2), H, W],...)
        <- Return
            clicks_for_interaction: np.array([int(obj_id#1), H, W], [int(obj_id#2), H, W],...)
        """
        negative_interactions[:,1:] *= self.img_res
        self.negative_clicks_for_interaction = negative_interactions


    def apply_positive_clicks(self, clicks):
        '''
            click: np.array([Obj_id,H,W])
        '''
        flag = "positive"
        for click in clicks:
            self.add_click(flag, click[1:])


    def apply_negative_clicks(self, clicks):
        '''
            click: np.array([Obj_id,H,W])
        '''
        flag   = "negative"
        for click in clicks:
            self.add_click(flag, click[1:])


    def add_click(self, flag, coords):
        '''
        -> Add a click to the Clicker instance
            flag: "positive"/"negative", True for positive click, False for negative click
            coords: np.array[H,W]
        '''
        click_is_positive = True if flag is "positive" else False
        self.inter_mod_clicker.add_click(Click(is_positive=click_is_positive, coords=coords))


    def add_padding_to_mask(self, mask_input_propagation_OG):
        '''
        -> Add a padding to predicted mask from the interaction module, in order to be integrated with the propagation module
            mask_input_propagation_OG: Tensor.Size([1,1,H,W])
        <- Return:
            mask_input_propagation: Tensor.Size([2,1,H+padding,W+padding]), want a background mask for MiVOS
        '''
        mask_input_propagation = torch.clone(mask_input_propagation_OG)
        if self.use_soft_switch: # On soft mask modus already did the aggregation of the background
            mask_input_propagation = pad_divide_by(mask_input_propagation, 16, mask_input_propagation.shape[-2:])[0]
        else:
            # Padding Mask / Aggreatating positive and negative masks
            mask_input_propagation = pad_divide_by(mask_input_propagation, 16, mask_input_propagation.shape[-2:])[0]  # Padding mask
            background_mask  = torch.sum(mask_input_propagation, dim=0) == 0
            background_mask  = background_mask.clone().detach().requires_grad_(False).unsqueeze(0).type(dtype=torch.float32)
            mask_input_propagation = torch.cat((background_mask, mask_input_propagation), dim=0)
        #-
        return mask_input_propagation


    def trim_padding_of_mask(self, propagated_masks):
        '''
        -> Trim the padding needed by for the propagation predictions back to the normal size
            propagated_masks: np.array([#frames, H+padding, W+padding])
        <- Return
            propagated_masks: np.array([#frames, H, W])
        '''
        if self.pad[2] + self.pad[3] > 0:
            propagated_masks = propagated_masks[:, self.pad[2]:-self.pad[3], :]
        if self.pad[0] + self.pad[1] > 0:
            propagated_masks = propagated_masks[:, :, self.pad[0]:-self.pad[1]]
        #-
        return propagated_masks


    def trim_padding_of_soft_mask(self, propagated_masks):
        '''
        -> Trim the padding needed by for the propagation predictions back to the normal size
            propagated_masks: np.array([#objects, #frames, H+padding, W+padding])
        <- Return
            propagated_masks: np.array([#objects, #frames, H, W])
        '''
        if self.pad[2] + self.pad[3] > 0:
            propagated_masks = propagated_masks[:, :, :, self.pad[2]:-self.pad[3], :]
        if self.pad[0] + self.pad[1] > 0:
            propagated_masks = propagated_masks[:, :, :, :, self.pad[0]:-self.pad[1]]
        #-
        return propagated_masks


    def update_propagated_masks_memory(self, propagated_masks):
        '''
            propagated_masks: np.array([#frames, H, W])
        '''
        self.propagated_masks = propagated_masks


    def extract_mask_from_propagation(self, Obj_id: int):
        '''
        -> Extract a mask from the propagation masks, in order to use it as a previous mask
            Obj_id: int
            self.propagated_masks: [#frames, 1, H, W]
        <- Return
            mask_from_propagation: # Shape [1,1,h,w]
        '''
        mask_from_propagation_0 = (self.propagated_masks[self.anno_f_idx] == Obj_id).astype(np.float32)
        mask_from_propagation_1 = np.expand_dims(mask_from_propagation_0, axis=0)
        mask_from_propagation   = np.expand_dims(mask_from_propagation_1, axis=0)
        #-
        return mask_from_propagation


    def extract_soft_mask_from_propagation(self, Obj_id: int):
        '''
        -> Extract a mask from the propagation masks, in order to use it as a previous mask
            Obj_id: int
            self.propagated_masks: [#objects, #frames, 1, H, W]
        <- Return
            soft_mask_from_propagation: # Shape [1,1,h,w]
        '''
        soft_mask_from_propagation = self.propagated_masks[Obj_id][self.anno_f_idx][0]
        soft_mask_from_propagation = np.expand_dims(np.expand_dims(soft_mask_from_propagation, axis=0), axis=0)
        #-
        return soft_mask_from_propagation


    def use_a_pseudo_prev_mask_from_prop(self, Obj_id: int):
        '''
        -> Use the propagated mask by propagation for the current frame as baseline for the previous mask
        OBJ_id: int
        '''
        if self.use_previous_mask_switch:
            if 0 != self.interacted_count:  # Can only use previous mask if propagation has propagated at least once
                if self.use_soft_in_prev_mask_switch:
                    # Use Soft Mask as prev. mask in interaction module
                    self.pseudo_prev_mask = self.extract_soft_mask_from_propagation(Obj_id)
                else:
                    # Use Hard Mask as prev. mask in interaction module
                    self.pseudo_prev_mask = self.extract_mask_from_propagation(Obj_id) # Extract soft/hard mask -H

                # Set pseudo previous (soft/hard) mask
                self.set_previous_mask_in_interaction_module(self.pseudo_prev_mask)


    def mask_prediction_by_the_interaction_module(self):
        '''
        -> Make segmentation masks prediction for all objects in the current frame by leveraging user inputs (clicks)
        <- Return
            mask_propagation_input: Tensor.Size([1,1,H+padding,W+padding]), (First mask is the background mask)
        '''
        # ------------------------------------------------------------------------------------
        # - View mask for the current annotated frame that was predicted in a previous round -
        if self.Visualization_switch and self.propagated_masks.shape != np.array([0]).shape:
            txt = 'Before interaction module'
            if self.use_soft_in_prev_mask_switch and self.interacted_count != 0:
                self.Visualize_soft_mask(self.propagated_masks, [self.anno_f_idx], txt=txt)
            else:
                self.Visualization(self.propagated_masks[self.anno_f_idx], txt=txt)


        positive_clicks = self.positive_clicks_for_interaction.astype(dtype=np.int32)
        negative_clicks = self.negative_clicks_for_interaction.astype(dtype=np.int32)


        # ---------------------------------
        # - Interaction Module Prediction -
        for Obj_id in self.indices_of_obj_in_sequence:
            self.use_a_pseudo_prev_mask_from_prop(Obj_id)
            # --> Apply Positive Clicks
            positive_obj_id_clicks = positive_clicks[positive_clicks[:,0] == Obj_id]
            if len(positive_obj_id_clicks) != 0:
                self.apply_positive_clicks(positive_obj_id_clicks)
            # --> Apply Negative Clicks
            negative_obj_id_clicks = negative_clicks[negative_clicks[:,0] == Obj_id]
            if len(negative_obj_id_clicks) != 0:
                self.apply_negative_clicks(negative_obj_id_clicks)

            self.interaction_predictionn(Obj_id)
            self.reset_predictor_n_clicker()

        # ------------------------------------------------
        # - Aggregate Results for the Propagation Module -
        if 1 == self.total_nbr_of_obj_in_sequence:
            self.mask_interaction_output = aggregate_wbg(self.mask_interaction_output, keep_bg=True,
                                                  hard=True) if self.use_soft_switch else self.mask_interaction_output
        else:
            self.mask_interaction_output = aggregate_wbg_vujas(self.mask_interaction_output,
                                                        keep_bg=True,
                                                        hard=True,
                                                        aggregation_vujas = self.agg_vujas) if self.use_soft_switch else self.mask_interaction_output

        # -----------------------------------------------------------------------------------------------
        # - Adapt Resolution of the Mask Predicted by the Interaction Module for the Propagation Module -
        mask_propagation_input = self.add_padding_to_mask(self.mask_interaction_output)


        # -------------------------------
        # - View Predicted Mask by interaction the module -
        if self.Visualization_switch:
            self.Visualization(mask_propagation_input,
                               txt='after interaction module')
        #-
        return mask_propagation_input


    def CiVOS_interact(self):
        '''
        -> CiVOS : Interaction + Propagation
        <- Return
            propagated_masks: np.array([#frames,H,W]), Obj_id's are used as values for the masks
            next_interact         : None/int, next frame to use for the interaction
            self.anno_f_idx       : int, current frame used for the interaction
        '''
        # ----------------------------------------
        # - Predict Mask for the Annotated Frame -
        interaction_prediction_adapted = self.mask_prediction_by_the_interaction_module()

        # ------------------------------------------
        # - Propagate Mask to the Remaining Frames -
        if self.use_soft_in_prev_mask_switch:
            # --> Use Soft Mask
            propagated_masks, propagated_soft_masks = self.propagation.interact_can_output_soft_masks_vujas(
                interaction_prediction_adapted, self.anno_f_idx)

            propagated_masks      = self.trim_padding_of_mask(propagated_masks)
            propagated_soft_masks = self.trim_padding_of_soft_mask(propagated_soft_masks)

            self.update_propagated_masks_memory(propagated_soft_masks)
        else:
            # --> Use Hard Mask
            propagated_masks = self.propagation.interact(interaction_prediction_adapted, self.anno_f_idx)
            propagated_masks = self.trim_padding_of_mask(propagated_masks)
            self.update_propagated_masks_memory(propagated_masks)



        next_interact          = None
        self.interacted_count += 1
        #-
        return propagated_masks, next_interact, self.anno_f_idx


# ----------------------------------------------------------------------------------------------------------------------
# =============== BELOW ARE ONLY FUNCTIONS FOR HELPING UNDERSTAND WHAT IS GOING ON THROUGH VISUALIZATION ===============
# ----------------------------------------------------------------------------------------------------------------------
    def Visualize_soft_mask(self, predicted_masks, frame_nbr, txt='None', alpha=0.5):
        '''
        --> Visualization support for the soft mask
            predicted_masks: np.array(#obj, #frames, 1, H, W)
            frame_nbr      : list[int_frame]
            txt            : str, to display on the window
            alpha          : float, transparency coefficient
        '''
        colors = [[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                  [0, 0, 0]]

        # Get image from buffer
        img_arr = self.Ori_img.copy()[:,:,::-1]
        img_arr = (img_arr/255).astype(np.float32)

        for obj_idx, _ in enumerate(predicted_masks):
            img_arr_copy = img_arr.copy().transpose(2, 0, 1)

            obj_mask       = predicted_masks[obj_idx][frame_nbr].squeeze() # obj_mask = np.array(H,W)
            obj_mask_3_dim = np.array([obj_mask, obj_mask, obj_mask])
            obj_mask_2_img = (obj_mask_3_dim.transpose(1, 2, 0)*colors[obj_idx]).transpose(2, 0, 1)
            for jdx in range(0, img_arr_copy.shape[0]):
                img_arr_copy[jdx, :, :] = img_arr_copy[jdx, :, :] * (1 - alpha) + obj_mask_2_img[jdx, :, :] * alpha

            img_arr_copy = img_arr_copy*255

            while True:
                cv2.imshow(txt, img_arr_copy.transpose(1, 2, 0).astype(np.uint8))
                key = cv2.waitKey(0)
                if key == ord('q'):
                    # cv2.destroyAllWindows()
                    cv2.destroyWindow(txt)
                    break


    def Add_Mask_and_Image_Visu(self, predicted_masks, idx, alpha=0.5):
        if torch.is_tensor(predicted_masks):
            # mask_arr = (mask[0].clone().squeeze().cpu().numpy() - 1)*(-1)  # inverse the background mask to get the foreground
            # mask_arr = (mask[1].clone().squeeze().cpu().numpy())
            mask_arr = predicted_masks.clone().squeeze().cpu().numpy()
        else:
            mask = predicted_masks
            mask_arr = np.zeros((mask.max() + 1, mask.shape[0], mask.shape[1]))
            for obj_id in range(0, mask.max() + 1):
                mask_arr[obj_id, :, :] = (mask == obj_id).astype(dtype=np.uint8)
            torch.tensor(mask_arr)
            mask_arr = pad_divide_by(torch.tensor(mask_arr), 16, mask_arr.shape[-2:])[0].numpy().astype(dtype=np.int8)

        # Display the image
        img = self.propagation.get_image_buffered(idx)
        img_arr = img.clone().squeeze().cpu().numpy()
        img_arr -= img_arr.min()
        img_arr = (img_arr * 255) / img_arr.max()
        img_arr = img_arr[::-1]

        img_zeros = np.zeros_like(img_arr)

        # Draw mask on image
        colors = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                  [255, 255, 255]]
        for m, color in zip(mask_arr, colors):
            for jdx, clr in enumerate(color):
                img_zeros[jdx, :, :] = img_zeros[jdx, :, :] + m * clr

        for jdx in range(0, img_zeros.shape[0]):
            img_arr[jdx, :, :] = img_arr[jdx, :, :] * (1 - alpha) + img_zeros[jdx, :, :] * alpha

        return img_arr


    def Visualization(self, predicted_masks, txt=None):
        img_arr = self.Add_Mask_and_Image_Visu(predicted_masks, self.anno_f_idx, alpha=0.5)

        while True:
            cv2.imshow('IMG', img_arr.transpose(1, 2, 0).astype(np.uint8))
            if txt != None:
                print(txt)
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyWindow('IMG')
                break


    def Visu_propagation(self, named_Win, predicted_masks, alpha=0.5, v_time=2):
        for idx in range(0, predicted_masks.shape[0]):  # Loop through the frames
            img_arr = self.Add_Mask_and_Image_Visu(predicted_masks[idx], idx, alpha)
            img_arr = np.ascontiguousarray(img_arr.astype(np.uint8).transpose(1, 2, 0))

            # Add frame number
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(img_arr, "#{0:04d}".format(idx), (5, img_arr.shape[0] - 5), font, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)

            cv2.imshow(named_Win, img_arr)
            key = cv2.waitKey(v_time)
            if key == ord('q'):
                break

        cv2.destroyWindow(named_Win)