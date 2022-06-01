# Click Generating Strategies (CGSies)
#
# by Stéphane Vujasinovic

import numpy as np
import cv2

from click_davisinteractive import utils as davis_utils

from my_utils.First_click_simulation import simulate_clicks_from_mask
from my_utils.depth_first_search import depth_first_search
from my_utils.Mask_diff import mask_diff

# from my_utils.Scribbles_2_Point import strategy_f1, strategy_f2


class Click_Gen_Strat:
    def __init__(self):
        self.dfs = depth_first_search()
        self.Clicker_from_mask = simulate_clicks_from_mask(self.dfs)

    def set_strat(self, strat: int):
        self.strat = strat

    def set_visualization(self, visu_flag: bool, Display:classmethod):
        self.visu_flag = visu_flag
        self.Display = Display

    def is_new_sequence(self, new_seq: bool):
        self.new_sequence = new_seq

    def set_annotate_frame(self, annotated_frame: int):
        self.annotated_frame = annotated_frame

    def get_GT_mask(self, gt_mask_davis_sequence):
        self.GT_mask = gt_mask_davis_sequence[self.annotated_frame]
        self.img_res = self.GT_mask.shape[-2:]

    def select_CGSy(
        self,
        scribbles,
        pred_masks,
        minimal_region_size=0,
        limit_for_points=3,
        minimal_area_to_considere=0,
    ):
        if 1 == self.strat:
            Generated_clicks = self.strategy_f1(scribbles, self.annotated_frame)
        elif 4 == self.strat:
            Generated_clicks = self.strategy_f1_bis(scribbles, self.annotated_frame, self.img_res)
        elif 2 == self.strat:
            Generated_clicks = self.strategy_f2(scribbles, self.annotated_frame,
                                                self.img_res, limit_for_points)
        elif 3 == self.strat:
            Generated_clicks = self.strategy_f3(
                pred_masks, minimal_region_size, limit_for_points, minimal_area_to_considere
            )
        return Generated_clicks

    def strategy_f3(
        self,
        pred_masks,
        minimal_region_size=0,
        limit_for_points=3,
        minimal_area_to_considere=0,
    ):

        # Store all positive and negative clicks to come
        total_positive_clicks = np.ones((1, 3)) * -1
        total_negative_clicks = np.ones((1, 3)) * -1

        if not self.new_sequence:
            # When interaction already occured - DANS UNE FONCTION CLick generating stategy
            PRED_mask_f      = pred_masks[self.annotated_frame]
            FN_mask, FP_mask = mask_diff(self.GT_mask, PRED_mask_f)

            # Generate clicks for the positive and negative interactions
            for obj_id in range(0, self.GT_mask.max()):
                FN_obj_mask, FP_obj_mask = FN_mask[obj_id], FP_mask[obj_id]

                # Generate positive clicks for Obj --> TODO avoir dans une fonction appart qui genere les clicks en fonction de mes strats
                clicks_d_pos, topographique_maps_pos = self.generate_clicks_f3(
                    FN_obj_mask,
                    minimal_region_size,
                    limit_for_points,
                    minimal_area_to_considere,
                )
                # Generate negative clicks for Obj
                clicks_d_neg, topographique_maps_neg = self.generate_clicks_f3(
                    FP_obj_mask,
                    minimal_region_size,
                    limit_for_points,
                    minimal_area_to_considere,
                )

                if 0 != len(clicks_d_pos):
                    nbr_of_positive_inter = (
                        np.ones((clicks_d_pos["Obj_id_1"].shape[0], 1)) * (obj_id + 1)
                    ).astype(np.int)
                    apdapted_clicks = np.concatenate(
                        (nbr_of_positive_inter, clicks_d_pos["Obj_id_1"]), axis=1
                    )
                    total_positive_clicks = np.concatenate(
                        (total_positive_clicks, apdapted_clicks), axis=0
                    )
                if 0 != len(clicks_d_neg):
                    nbr_of_negative_inter = (
                        np.ones((clicks_d_neg["Obj_id_1"].shape[0], 1)) * (obj_id + 1)
                    ).astype(np.int)
                    apdapted_clicks = np.concatenate(
                        (nbr_of_negative_inter, clicks_d_neg["Obj_id_1"]), axis=1
                    )
                    total_negative_clicks = np.concatenate(
                        (total_negative_clicks, apdapted_clicks), axis=0
                    )

        else:
            # In the case where a new sequence is initiated, only one element to annotate
            clicks_d, topographique_maps = self.generate_clicks_f3(
                self.GT_mask, minimal_region_size, limit_for_points
            )

            keys = clicks_d.keys()

            for obj_id, key in enumerate(keys):  # range(0,GT_mask_f.max()):
                FN_obj_mask = (self.GT_mask == obj_id + 1).astype(np.int)

                nbr_of_positive_inter = (
                    np.ones((clicks_d[key].shape[0], 1)) * (obj_id + 1)
                ).astype(np.int)
                apdapted_clicks = np.concatenate(
                    (nbr_of_positive_inter, clicks_d[key]), axis=1
                )
                total_positive_clicks = np.concatenate(
                    (total_positive_clicks, apdapted_clicks), axis=0
                )



        # Keep the clicks needed
        total_positive_clicks = total_positive_clicks[1:]
        total_negative_clicks = total_negative_clicks[1:]



        #------------------------------------ DANS UNE FONCTION
        # Visualization TODO avoir dans une fonction a part
        if self.visu_flag:
            print('--> FN:', total_positive_clicks)
            print('--> FP:', total_negative_clicks)
            print(self.annotated_frame)
            self.Display.read_frame_anno(self.annotated_frame)
            colors_map = self.Display.cmap
            GT_mask_to_show = np.zeros((self.GT_mask.shape[0], self.GT_mask.shape[1], 3))
            FN_mask_to_show = np.zeros((self.GT_mask.shape[0], self.GT_mask.shape[1], 3))
            FP_mask_to_show = np.zeros((self.GT_mask.shape[0], self.GT_mask.shape[1], 3))
            Pred_mask_to_show = np.zeros((self.GT_mask.shape[0], self.GT_mask.shape[1], 3))

            for obj_id in range(1, self.GT_mask.max() + 1):
                GT_mask_to_show[self.GT_mask == obj_id] = colors_map[obj_id]

                positive_clicks = total_positive_clicks[:, 1:][total_positive_clicks[:, 0] == obj_id]
                negative_clicks = total_negative_clicks[:, 1:][total_negative_clicks[:, 0] == obj_id]

                if len(positive_clicks) != 0:
                    for jdx in range(-1, 2):
                        for idx in range(-1, 2):
                            GT_mask_to_show[(positive_clicks[:, 0] - jdx).astype(np.int),
                            (positive_clicks[:, 1] - idx).astype(np.int), :] = [0, 255, 0]
                if len(negative_clicks) != 0:
                    for jdx in range(-1, 2):
                        for idx in range(-1, 2):
                            GT_mask_to_show[(negative_clicks[:, 0] - jdx).astype(np.int),
                            (negative_clicks[:, 1] - idx).astype(np.int), :] = [0, 0, 255]

                if not self.new_sequence:
                    FN_mask_to_show[FN_mask[obj_id - 1] == 1] = colors_map[obj_id]
                    FP_mask_to_show[FP_mask[obj_id - 1] == 1] = colors_map[obj_id]
                    Pred_mask_to_show[pred_masks[self.annotated_frame] == obj_id] = colors_map[obj_id]

            if self.new_sequence:
                global_topo_graph = np.sum(topographique_maps, axis=0).astype(np.int8)
            else:
                global_topo_graph_pos = np.sum(topographique_maps_pos, axis=0).astype(np.int8)
                global_topo_graph_neg = np.sum(topographique_maps_neg, axis=0).astype(np.int8)

            while True:
                cv2.imshow('GT_mask', GT_mask_to_show.astype(np.int8))
                if not self.new_sequence:
                    cv2.imshow("topographique_maps_pos", global_topo_graph_pos)
                    cv2.imshow("topographique_maps_neg", global_topo_graph_neg)
                    cv2.imshow('FN_mask', FN_mask_to_show.astype(np.int8))
                    cv2.imshow('FP_mask', FP_mask_to_show.astype(np.int8))
                    cv2.imshow('Pred_mask', Pred_mask_to_show.astype(np.int8))
                else:
                    cv2.imshow("topographique_maps", global_topo_graph)


                key = cv2.waitKey(0)

                if ord('q') == key:
                    break

            self.Display.show_image()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Visualization --> END
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #-
        return total_positive_clicks, total_negative_clicks


    def generate_clicks_f3(
        self,
        mask,
        minimal_region_size=0,
        limit_for_points=3,
        minimal_area_to_considere=0,
    ):
        topographique_maps = self.Clicker_from_mask.create_topographique_maps(
            mask, minimal_region_size
        )
        clicks_d = self.Clicker_from_mask.find_clicks_center(
            mask, topographique_maps, limit_for_points, minimal_area_to_considere
        )

        return clicks_d, topographique_maps



    def extract_a_point_from_scribbles(self, obj_points):
        # Extract central point
        yx_mean = np.mean(obj_points, axis=0)

        # Find shortes distante point and draw it
        dist_yx = (yx_mean - obj_points) ** 2
        dist_yx = np.sum(dist_yx, axis=1)

        return dist_yx.argmin(), yx_mean


    def strategy_f1(self, scribbles, frame_anno):
        '''Infer only one click for all scribbles belonging to the same instance'''
        points_2_draw_assemble = scribbles['scribbles'][frame_anno]
        points_2_assemble = np.empty((0, 3))
        for elem_2_assemble in points_2_draw_assemble:
            obj_id = elem_2_assemble['object_id']
            obj_points = np.array(elem_2_assemble['path'])
            obj_id_vec = np.ones((obj_points.shape[0], 1)) * obj_id
            compose_points_with_id = np.concatenate((obj_id_vec, obj_points), axis=1)
            points_2_assemble = np.append(points_2_assemble, compose_points_with_id, axis=0)

        assembled_points_d = {}
        for obj_id in np.unique(points_2_assemble[:, 0]).astype(np.int):
            key_n = "Obj_id_{}".format(obj_id)
            points_to_assemble = points_2_assemble[obj_id == points_2_assemble[:, 0], :]
            assembled_points_d[key_n] = points_to_assemble[:, 1::]


        positive_clicks = np.ones((1, 3)) * -1
        negative_clicks = np.ones((1, 3)) * -1
        inferred_clicks_regardless_if_pos_or_neg = np.ones((1, 3)) * -1
        for key in assembled_points_d.keys():
            obj_id     = int(key.split("_")[-1])
            obj_points = assembled_points_d[key]

            # Check if there are duplicated
            _, counts = np.unique(obj_points, return_counts=True, axis=1)

            # Extract central point
            closest_points, yx_mean = self.extract_a_point_from_scribbles(obj_points)

            closest_2_mean_point = obj_points[closest_points][::-1]

            inferred_clicks_regardless_if_pos_or_neg = np.concatenate((inferred_clicks_regardless_if_pos_or_neg,
                                                                       np.array([[obj_id,*closest_2_mean_point]])),
                                                                       axis = 0)
            if obj_id != 0:
                positive_clicks = np.concatenate((positive_clicks,
                                                  np.array([[obj_id,*closest_2_mean_point]])), axis = 0)

        inferred_clicks_regardless_if_pos_or_neg = inferred_clicks_regardless_if_pos_or_neg[1:]

        if 'Obj_id_0' in assembled_points_d.keys():
            for key in assembled_points_d.keys():
                obj_id = int(key.split("_")[-1])

                if obj_id == 0:
                    continue
                else:
                    negative_clicks_candidates = inferred_clicks_regardless_if_pos_or_neg[inferred_clicks_regardless_if_pos_or_neg[:, 0] != obj_id][:,-2:]
                    for candidate in negative_clicks_candidates:
                        negative_clicks = np.concatenate((negative_clicks,
                                                          np.array([[obj_id,*candidate[-2:]]])),
                                                         axis=0)

        positive_clicks = positive_clicks[1:]
        negative_clicks = negative_clicks[1:]

        print(positive_clicks)
        print(negative_clicks)

        return positive_clicks, negative_clicks


    def compute_central_coord(self, mask, obj_id):
        h_coords, w_coords = np.where(mask == obj_id)
        h_mean, w_mean = np.mean((h_coords, w_coords), axis=1)
        h_dist, w_dist = np.power(h_coords - h_mean, 2), np.power(w_coords - w_mean, 2)
        dist = h_dist + w_dist
        shortest_dist = np.argmin(dist)
        pnt_h_coord, pnt_w_coord = h_coords[shortest_dist], w_coords[shortest_dist]

        return pnt_h_coord, pnt_w_coord


    def strategy_f1_bis(self, scribbles, frame_anno, H_W_img):
        '''Infer a click per scribble'''
        mask = davis_utils.scribbles.scribbles2mask(scribbles, H_W_img)
        current_annotated_mask = mask[frame_anno]

        List_of_obj_ids = []
        for elem in scribbles['scribbles'][frame_anno]:
            List_of_obj_ids.append(elem['object_id'])

        positive_clicks = np.ones((1, 3)) * -1
        negative_clicks = np.ones((1, 3)) * -1
        inferred_clicks_regardless_if_pos_or_neg = np.ones((1, 3)) * -1

        # keys = Dict_with_seperated_scribbles_instances.keys()
        for obj_id in np.unique(current_annotated_mask):
            if obj_id == -1:
                continue

            pnt_h_coord, pnt_w_coord = self.compute_central_coord(current_annotated_mask, obj_id)
            pnt_h_coord /= H_W_img[0]
            pnt_w_coord /= H_W_img[1]

            inferred_clicks_regardless_if_pos_or_neg = np.concatenate((inferred_clicks_regardless_if_pos_or_neg,
                                                                       np.array([[obj_id,
                                                                                  pnt_h_coord,
                                                                                  pnt_w_coord]])), axis = 0)

        inferred_clicks_regardless_if_pos_or_neg = inferred_clicks_regardless_if_pos_or_neg[1::]
        for elem in inferred_clicks_regardless_if_pos_or_neg:
            obj_id = elem[0]
            if 0 == obj_id:
                continue

            positive_clicks = np.concatenate((positive_clicks, np.array([[obj_id, *elem[-2:]]])), axis=0)
            negative_clicks_candidates = inferred_clicks_regardless_if_pos_or_neg[inferred_clicks_regardless_if_pos_or_neg[:, 0] != obj_id][:, -2:]

            if 0 in List_of_obj_ids:    # Only if background click start to incorporate negative interactions
                for candidate in negative_clicks_candidates:
                    negative_clicks = np.concatenate((negative_clicks,
                                                      np.array([[obj_id, *candidate]])),
                                                     axis=0)

        positive_clicks = positive_clicks[1:]
        negative_clicks = negative_clicks[1:]

        print(positive_clicks)
        print(negative_clicks)

        return positive_clicks, negative_clicks



    def strategy_f2(self, scribbles, frame_anno, H_W_img, limit_for_points = 3):
        '''Infer a click per scribble'''
        mask = davis_utils.scribbles.scribbles2mask(scribbles, H_W_img)
        current_annotated_mask = mask[frame_anno]

        dfs = depth_first_search()
        Dict_with_seperated_scribbles_instances = dfs.extract_scribbles_properly(current_annotated_mask)
        # print(Dict_with_seperated_scribbles_instances.keys())

        keys = Dict_with_seperated_scribbles_instances.keys()

        Dict_with_points = {}
        # means_calculated = []
        Ori_coords = []

        positive_clicks = np.ones((1, 3)) * -1
        negative_clicks = np.ones((1, 3)) * -1
        inferred_clicks_regardless_if_pos_or_neg = np.ones((1, 3)) * -1

        for key in keys:
            obj_id        = int(key.split("_")[-1])
            scribble_mask = Dict_with_seperated_scribbles_instances[key]
            seperate_scribble_idx = np.arange(scribble_mask.min(), scribble_mask.max())+1
            points_to_store_in_multiple_clicks = []
            for scribble_idx in seperate_scribble_idx[:limit_for_points]:
                pnt_h_coord, pnt_w_coord = self.compute_central_coord(scribble_mask, scribble_idx)
                # h_coords, w_coords = np.where(scribble_mask == scribble_idx)
                #
                # h_mean, w_mean = np.mean((h_coords, w_coords), axis = 1)
                # h_dist, w_dist = np.power(h_coords - h_mean,2), np.power(w_coords - w_mean,2)
                # dist           = h_dist + w_dist
                # shortest_dist  = np.argmin(dist)
                # pnt_h_coord, pnt_w_coord = h_coords[shortest_dist], w_coords[shortest_dist]

                Ori_coords.append(np.array([pnt_h_coord,pnt_w_coord]))

                pnt_h_coord /= H_W_img[0]
                pnt_w_coord /= H_W_img[1]
                points_to_store_in_multiple_clicks.append([np.array([pnt_h_coord,pnt_w_coord])])
                # means_calculated.append([h_mean/H_W_img[0], w_mean/H_W_img[1]])

                inferred_clicks_regardless_if_pos_or_neg = np.concatenate((inferred_clicks_regardless_if_pos_or_neg,
                                                                           np.array([[obj_id, pnt_h_coord, pnt_w_coord]])), axis=0)

                if obj_id != 0:
                    positive_clicks = np.concatenate((positive_clicks,
                                                      np.array([[obj_id, pnt_h_coord, pnt_w_coord]])), axis=0)

            Dict_with_points[key] = points_to_store_in_multiple_clicks

        inferred_clicks_regardless_if_pos_or_neg = inferred_clicks_regardless_if_pos_or_neg[1:]

        # if 'Obj_id_0' in keys:
        #     for key in keys:
        #         obj_id = int(key.split("_")[-1])
        #
        #         if obj_id == 0: # Don't make negative clicks for the background, skip negative clicks for the background
        #             continue
        #         else:
        #             negative_clicks_candidates = inferred_clicks_regardless_if_pos_or_neg[inferred_clicks_regardless_if_pos_or_neg[:, 0] != obj_id][:,-2:]
        #             for candidate in negative_clicks_candidates:
        #                 negative_clicks = np.concatenate((negative_clicks,
        #                                                   np.array([[obj_id,*candidate]])),
        #                                                  axis=0)

        for key in keys:
            obj_id = int(key.split("_")[-1])

            if obj_id == 0: # Don't make negative clicks for the background, skip negative clicks for the background
                continue
            else:
                negative_clicks_candidates = inferred_clicks_regardless_if_pos_or_neg[inferred_clicks_regardless_if_pos_or_neg[:, 0] != obj_id][:,-2:]
                for candidate in negative_clicks_candidates:
                    negative_clicks = np.concatenate((negative_clicks,
                                                      np.array([[obj_id,*candidate]])),
                                                     axis=0)

        positive_clicks = positive_clicks[1:]
        negative_clicks = negative_clicks[1:]

        print(positive_clicks)
        print(negative_clicks)

        return positive_clicks, negative_clicks


