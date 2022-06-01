# Simulate de First Click based on the original paper FCANet
#
# by Stéphane Vujasinovic

import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import time

class simulate_clicks_from_mask():
    def __init__(self, dfs):
        self.dfs = dfs

    def border_of_mask(self, mask_pad):
        dillated_mask    = binary_dilation(mask_pad).astype(mask_pad.dtype)
        border_mask      = dillated_mask - mask_pad
        return border_mask

    def create_empty_array(self, mask_pad, border_mask):
        Nbr_p_border_mask = np.sum(border_mask)
        D_Mask = np.ones((np.shape(mask_pad)[0], np.shape(mask_pad)[1], Nbr_p_border_mask), dtype=np.float32)
        return D_Mask

    def distances(self, mask_pad, border_mask, D_Mask, mask_3D):

        y_border, x_border = np.where(border_mask>0)   # Extract y,x coordinates for border

        # Create y coord mask
        Y_axis   = np.expand_dims(np.arange(0,np.shape(mask_pad)[0]), axis = 0)
        Y_axis_m = np.ones([np.shape(mask_pad)[0],np.shape(mask_pad)[1]])
        Y_mask   = Y_axis_m*Y_axis.T
        m_Y = D_Mask * np.expand_dims(Y_mask, axis = 2)
        # Create x coord mask
        X_axis   = np.expand_dims(np.arange(0,np.shape(mask_pad)[1]), axis = 0)
        X_axis_m = np.ones([np.shape(mask_pad)[0],np.shape(mask_pad)[1]])
        X_mask   = X_axis_m*X_axis
        m_X = D_Mask * np.expand_dims(X_mask, axis = 2)

        # Create x and y mask containing border coordinates
        Y_2D = np.expand_dims(y_border, axis = 0)
        Y_3D = np.expand_dims(Y_2D, axis = 1)     # Now I have y coord. of the border on the channels
        Y_border_mask    = Y_3D*D_Mask
        Y_minus_Y_border = m_Y-Y_border_mask
        X_2D = np.expand_dims(x_border, axis = 0)
        X_3D = np.expand_dims(X_2D, axis = 1)
        X_border_mask    = X_3D*D_Mask
        X_minus_X_border = m_X-X_border_mask

        #
        Y_minus_Y_border*=mask_3D
        X_minus_X_border*=mask_3D

        # Now Euclidean distance
        sq_Y_minus_Y_border = Y_minus_Y_border*Y_minus_Y_border
        sq_X_minus_X_border = X_minus_X_border*X_minus_X_border

        sq_dist_X_Y = sq_Y_minus_Y_border+sq_X_minus_X_border

        minim_sq_dist = np.min(sq_dist_X_Y, axis = 2)

        return minim_sq_dist

    def max_dist_and_point_coord_of_center(self, minim_sq_dist):
        maximum_distance_2_Neg_region = np.max(minim_sq_dist)
        coord_y, coord_x = np.where(minim_sq_dist == maximum_distance_2_Neg_region)     # Point where E(d) = 1
        coord_point_center = np.array((np.average(coord_y)-1, np.average(coord_x)-1), dtype = np.int32)

        return coord_point_center, maximum_distance_2_Neg_region


    def find_first_click(self, mask_OG):
        mask_pad = np.pad(mask_OG.copy(), ((1, 1), (1, 1)), mode='constant')
        mask_3D  = np.expand_dims(mask_pad, axis = 2)
        border_mask   = self.border_of_mask((mask_pad).astype(np.int))
        D_Mask = self.create_empty_array(mask_pad, border_mask)
        minim_sq_dist = self.distances(mask_pad, border_mask, D_Mask, mask_3D)
        coord_point_center, maximum_distance_2_Neg_region = self.max_dist_and_point_coord_of_center(minim_sq_dist)
        # print("pause")

        return maximum_distance_2_Neg_region, coord_point_center


    def create_topographique_maps(self, mask_OG, minimal_region_size=None):
        """
        Finf clicks for each object
        :param mask:
        :return:
        """
        mask_pad = np.pad(mask_OG.copy(), ((1, 1), (1, 1)), mode='constant')

        topographique_maps = np.zeros([mask_OG.max(), mask_pad.shape[0], mask_pad.shape[1]])

        for m_idx in range(1, mask_OG.max() + 1):
            ero_mask  = (mask_pad==m_idx).astype(np.int)
            if minimal_region_size is None or 0 == minimal_region_size:
                go_until = 0
            else:
                go_until = int(ero_mask.sum()/minimal_region_size)

            topo_mask = ero_mask.copy()
            while go_until < ero_mask.sum():
                ero_mask  = binary_erosion(ero_mask).astype(mask_pad.dtype)
                topo_mask += ero_mask
                # print(ero_mask.sum())

            topographique_maps[m_idx-1] = topo_mask

        # unpadding
        topographique_maps = topographique_maps[:,1:-1, 1:-1]

        return topographique_maps

    def find_clicks_center(self, toy_example, topographique_maps, limit_of_points = 3, minimal_size_of_region_to_be_considered_as_click = 0):
        """
        Find the center for the regions in a deterministic way
        :param topographique_maps:
        :param limit_of_points:
        :return:
        """
        ordered_map = self.dfs.extract_scribbles_properly(toy_example)

        Dict_clicks = {}
        keys_click = ["Obj_id_{}".format(obj_id) for obj_id in range(1, topographique_maps.shape[0]+1)]
        for (topo_map, key_click) in zip(topographique_maps, keys_click):
            # Take only the maximum values
            map = (topo_map == topo_map.max()).astype(np.int)
            H_W_img = map.shape
            dfs_Dict = self.dfs.extract_scribbles_properly(map)

            keys = dfs_Dict.keys()

            # Dict_with_points = {}
            Ori_coords = []

            for key in keys:
                scribble_mask = dfs_Dict[key]
                seperate_scribble_idx = np.arange(scribble_mask.min(), scribble_mask.max()) + 1
                points_to_store_in_multiple_clicks = np.ones([1,2])*-1
                for scribble_idx in seperate_scribble_idx[:limit_of_points]:
                    h_coords, w_coords = np.where(scribble_mask == scribble_idx)

                    h_mean, w_mean = np.mean((h_coords, w_coords), axis=1)
                    h_dist, w_dist = np.power(h_coords - h_mean, 2), np.power(w_coords - w_mean, 2)
                    dist = h_dist + w_dist
                    shortest_dist = np.argmin(dist)
                    pnt_h_coord, pnt_w_coord = h_coords[shortest_dist], w_coords[shortest_dist]

                    Save_click = True

                    if minimal_size_of_region_to_be_considered_as_click != 0:
                        idx_to_check_if_area_big_enough = ordered_map[key_click][pnt_h_coord, pnt_w_coord]
                        area = (ordered_map[key_click] == idx_to_check_if_area_big_enough).sum()
                        if area < minimal_size_of_region_to_be_considered_as_click:
                            Save_click = False

                    if Save_click:
                        Ori_coords.append(np.array([pnt_h_coord, pnt_w_coord]))
                        pnt_h_coord /= H_W_img[0]
                        pnt_w_coord /= H_W_img[1]
                        points_to_store_in_multiple_clicks = np.append(points_to_store_in_multiple_clicks,np.expand_dims(np.array([pnt_h_coord, pnt_w_coord]),axis=0),axis = 0)

            Dict_clicks[key_click] = points_to_store_in_multiple_clicks[1:]

        return Dict_clicks


    def find_clicks_center_radius(self, toy_example, topographique_maps, radius = 5, limit_of_points = 3, minimal_size_of_region_to_be_considered_as_click = 0):
        """
        Find the center for the regions in a deterministic way
        :param topographique_maps:
        :param limit_of_points:
        :return:
        """
        ordered_map = self.dfs.extract_scribbles_properly(toy_example)
        Dict_clicks = {}
        keys_click = ["Obj_id_{}".format(obj_id) for obj_id in range(1, topographique_maps.shape[0]+1)]
        for (topo_map, key_click) in zip(topographique_maps, keys_click):
            # Take only the maximum values
            map = (topo_map == topo_map.max()).astype(np.int)
            H_W_img = map.shape
            dfs_Dict = self.dfs.extract_scribbles_properly(map)

            keys = dfs_Dict.keys()

            Dict_with_points = {}
            Ori_coords = []

            for key in keys:
                scribble_mask = dfs_Dict[key]
                seperate_scribble_idx = np.arange(scribble_mask.min(), scribble_mask.max()) + 1
                points_to_store_in_multiple_clicks = np.ones([1,2])*-1
                for scribble_idx in seperate_scribble_idx[:limit_of_points]:
                    h_coords, w_coords = np.where(scribble_mask == scribble_idx)

                    h_mean, w_mean = np.mean((h_coords, w_coords), axis=1)
                    h_dist, w_dist = np.power(h_coords - h_mean, 2), np.power(w_coords - w_mean, 2)
                    dist = h_dist + w_dist
                    shortest_dist = np.argmin(dist)
                    pnt_h_coord, pnt_w_coord = h_coords[shortest_dist], w_coords[shortest_dist]

                    Save_click = True

                    if minimal_size_of_region_to_be_considered_as_click != 0:
                        idx_to_check_if_area_big_enough = ordered_map[key_click][pnt_h_coord, pnt_w_coord]
                        area = (ordered_map[key_click] == idx_to_check_if_area_big_enough).sum()
                        if area < minimal_size_of_region_to_be_considered_as_click:
                            Save_click = False

                    if Save_click:
                        Ori_coords.append(np.array([pnt_h_coord, pnt_w_coord]))

                        # Compute r and theta
                        r     = np.random.rand()*(radius+1)-0.5
                        theta = np.random.rand()*2*np.pi

                        h_err = np.sin(theta)*r
                        w_err = np.cos(theta)*r

                        pnt_h_coord_err = int(pnt_h_coord + h_err)
                        pnt_w_coord_err = int(pnt_w_coord + w_err)

                        pnt_h_coord_err /= H_W_img[0]
                        pnt_w_coord_err /= H_W_img[1]

                        points_to_store_in_multiple_clicks = np.append(points_to_store_in_multiple_clicks,np.expand_dims(np.array([pnt_h_coord_err, pnt_w_coord_err]),axis=0),axis = 0)


            Dict_clicks[key_click] = points_to_store_in_multiple_clicks[1:]

        return Dict_clicks


    def find_clicks_random_in_region(self, toy_example, topographique_maps, limit_of_points=3, minimal_size_of_region_to_be_considered_as_click = 0):
        """
        Find a center for the regions in a random manner
        :param topographique_maps:
        :param limit_of_points:
        :return:
        """
        ordered_map = self.dfs.extract_scribbles_properly(toy_example)
        Dict_clicks = {}
        keys_click = ["Obj_id_{}".format(obj_id) for obj_id in range(1, topographique_maps.shape[0]+1)]
        for (topo_map, key_click) in zip(topographique_maps, keys_click):
            # Take only the maximum values
            map = (topo_map == topo_map.max()).astype(np.int)
            H_W_img = map.shape
            dfs_Dict = self.dfs.extract_scribbles_properly(map)

            keys = dfs_Dict.keys()

            Dict_with_points = {}
            Ori_coords = []

            for key in keys:
                scribble_mask = dfs_Dict[key]
                seperate_scribble_idx = np.arange(scribble_mask.min(), scribble_mask.max()) + 1
                points_to_store_in_multiple_clicks = np.ones([1,2])*-1
                for scribble_idx in seperate_scribble_idx[:limit_of_points]:
                    # Find ramdomly a point, define a seed before for reproducibility
                    h_coords, w_coords = np.where(scribble_mask == scribble_idx)

                    # from this set of coords select a random value randing from both sets
                    pnt_h_coord = np.random.choice(h_coords)
                    pnt_w_coord = np.random.choice(w_coords)

                    Save_click = True

                    if minimal_size_of_region_to_be_considered_as_click != 0:
                        idx_to_check_if_area_big_enough = ordered_map[key_click][pnt_h_coord, pnt_w_coord]
                        area = (ordered_map[key_click] == idx_to_check_if_area_big_enough).sum()
                        if area < minimal_size_of_region_to_be_considered_as_click:
                            Save_click = False

                    if Save_click:
                        Ori_coords.append(np.array([pnt_h_coord, pnt_w_coord]))
                        pnt_h_coord /= H_W_img[0]
                        pnt_w_coord /= H_W_img[1]

                        points_to_store_in_multiple_clicks = np.append(points_to_store_in_multiple_clicks,np.expand_dims(np.array([pnt_h_coord, pnt_w_coord]),axis=0),axis = 0)


            Dict_clicks[key_click] = points_to_store_in_multiple_clicks[1:]

        return Dict_clicks


# Test the module
if "__main__" == __name__:
    from depth_first_search import depth_first_search
    # define random seed
    np.random.seed(0)

    toy_ex = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0],
                       [0,1,1,1,1,0,0,0,0,0,0,0,3,3,3,0],
                       [0,0,1,1,1,1,0,0,0,0,0,0,3,3,3,0],
                       [0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,0],
                       [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                       [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,2,2,2,0,0,0,0,0,3,3,3,3,3,0,0],
                       [0,2,2,2,0,0,0,0,0,3,3,3,3,3,0,0],
                       [0,2,2,2,0,0,0,0,0,3,3,3,3,3,0,0]])

    # toy_ex = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    #                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    #                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    #                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    #                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])

    dfs = depth_first_search()
    click_finder = simulate_clicks_from_mask(dfs)

    topographique_maps = click_finder.create_topographique_maps(toy_ex)
    # dict_click_centers_A = click_finder.find_clicks_center(toy_ex, topographique_maps, limit_of_points=100)
    # dict_click_centers_B = click_finder.find_clicks_center(toy_ex, topographique_maps, limit_of_points=3)
    dict_click_centers_C = click_finder.find_clicks_center(toy_ex, topographique_maps, limit_of_points=100, minimal_size_of_region_to_be_considered_as_click = 13)            # Find centers

    dict_click_centers_bis = click_finder.find_clicks_center_radius(toy_ex, topographique_maps)  # Center with an error region around

    minimal_region_size = 2
    topographique_maps_4xsmaller = click_finder.create_topographique_maps(toy_ex, minimal_region_size)
    dict_click_regions_tres      = click_finder.find_clicks_random_in_region(toy_ex, topographique_maps_4xsmaller)  # Generate random center

    dict_click_centers_quatro = click_finder.find_clicks_center(toy_ex, topographique_maps_4xsmaller)            # Find centers

    dict_click_centers_cinco = click_finder.find_clicks_center_radius(toy_ex, topographique_maps_4xsmaller)  # Center with an error region around

    print("pause")