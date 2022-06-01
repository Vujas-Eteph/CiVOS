# Check if the scribbles are connected
#
#

import numpy as np

class depth_first_search():
    """
    My implementation of the depth first search
    """
    def __init__(self):
        # Check separate scribbles
        self.reset()
        self.set_kernel(3)

    def reset(self):
        """

        :return:
        """
        self.queue = np.ones([1,3])*-1
        self.nbr_of_separate_scribbles_for_same_instance = 1

    def set_kernel(self,kernel_size):
        """

        :param kernel_size:
        :return:
        """
        self.kernel = kernel_size


    def manage_queue_entries(self, possible_new_neighbor_nodes):
        """"""
        self.queue         = np.vstack((self.queue, possible_new_neighbor_nodes))   # Add to queue
        _, order_for_queue = np.unique(self.queue, axis = 0, return_index=True)     # Keep only unique coord
        self.queue         = self.queue[order_for_queue]


    def manage_queue_exists(self):
        """"""
        new_coords = self.queue[1:][-1].astype(np.int)
        self.queue = self.queue[:-1]
        return  new_coords[0], new_coords[1]


    def update_masks(self, h_pos,w_pos):
        """"""
        self.Mask_w_obj_id[h_pos,w_pos] = 0
        self.Mask_w_scribble_id[h_pos,w_pos] = self.nbr_of_separate_scribbles_for_same_instance

    def list_coord_to_highligh(self):
        """"""
        H_coords, W_coords = np.where(self.Mask_w_obj_id == 1)  # list coordinates of the highlighted elements
        return H_coords[0], W_coords[0]


    def node_to_visit_in_depth_first_search_wise(self, obj_id, h_pos, w_pos):
        """

        :param idx:
        :return:
        """
        # Check neighbors value
        neighbor_values = np.zeros([self.kernel*self.kernel, self.kernel])  # array [h_pos, w_pos, value] to be expected
        low_boundary    = int((1 - self.kernel) / 2)
        high_boundary   = int(1 + ((self.kernel - 1) / 2))
        filter_res      = - 1
        neigh_idx       = 0

        for hdx in range(low_boundary, high_boundary):
            for wdx in range(low_boundary, high_boundary):
                neighbor_values[neigh_idx] = np.array([hdx + h_pos, wdx + w_pos,
                                                       self.Mask_w_obj_id[hdx + h_pos, wdx + w_pos]])

                filter_res += self.Mask_w_obj_id[hdx + h_pos, wdx + w_pos]
                neigh_idx  +=1

        neighbor_values = np.vstack((neighbor_values[:4], neighbor_values[5:]))
        self.update_masks(h_pos,w_pos) # Set the value of visited nodes to 0, so that i don't revisit them


        if filter_res > 0:  # connected to a new node
            # check which nodes a connected to the current note
            neighbor_nodes = neighbor_values[neighbor_values[:,-1] == 1]
            self.manage_queue_entries(neighbor_nodes)
            next_h_node_coord, next_w_node_coord = self.manage_queue_exists()
            # Keep following depth first the search
            return obj_id, next_h_node_coord, next_w_node_coord

        else:
            if len(self.queue) == 1: # mean no more values in the queue
                # Check if all nodes have been visited
                if self.Mask_w_obj_id.max() != 0: # Not all nodes have been visited
                    self.nbr_of_separate_scribbles_for_same_instance +=1
                    next_h_node_coord, next_w_node_coord = self.list_coord_to_highligh()
                    return obj_id, next_h_node_coord, next_w_node_coord

                else:
                    return obj_id, False, False # Just return last obj_id but let the other values as is

            else:   # not all nodes in the queue have been visited... backtrace until the end
                next_h_node_coord, next_w_node_coord = self.manage_queue_exists()
                return obj_id, next_h_node_coord, next_w_node_coord


    def extract_scribbles_properly(self, array_w_scribbles):
        """
        Extract scribbles as one element if smaller scribbles compose it and are connected to one another
        input: array
        :return:
        """
        obj_ids = np.unique(array_w_scribbles)[1:]  #--> first element is expected to be the id for no scirbbles
        Dict = {}
        for obj_id in obj_ids:
            # print('OBJ_ID:', obj_id)
            self.obj_id = obj_id
            self.reset()

            self.Mask_w_obj_id_or   = (array_w_scribbles==obj_id).astype(np.int)                    # highlight an obj id in the array
            self.Mask_w_obj_id      = np.pad(self.Mask_w_obj_id_or.copy(),((1,1),(1,1)), 'constant', constant_values=((0,0),(0,0))) # Serves also as history for visited nodes
            self.Mask_w_scribble_id = np.zeros_like(self.Mask_w_obj_id)                             # Keep track of the scribbles belonging the same scribble id for the same obj id

            start_H_coord, start_W_coord = self.list_coord_to_highligh()

            while self.Mask_w_obj_id.max() != 0:
                # print('Max:',self.Mask_w_obj_id.sum())
                obj_id, start_H_coord, start_W_coord = self.node_to_visit_in_depth_first_search_wise(obj_id,
                                                                                                     start_H_coord,
                                                                                                     start_W_coord)

            Mask_w_scribble_id_unpad = self.Mask_w_scribble_id[1:-1,1:-1]

            Mask_w_scribble_id_unpad_ordered = self.order_scribbles_based_on_size(Mask_w_scribble_id_unpad)

            Dict["Obj_id_{}".format(obj_id)] = Mask_w_scribble_id_unpad_ordered

        return Dict


    def order_scribbles_based_on_size(self,Mask_w_scribble_id):
        new_order_array  = np.array([-1])
        max_scribble_idx = Mask_w_scribble_id.max()
        Mask_w_scribble_id_ordered = np.zeros_like(Mask_w_scribble_id)

        # Store size
        for idx in range(1,max_scribble_idx+1):
            nbr_of_elements = (Mask_w_scribble_id == idx).sum()
            new_order_array = np.hstack((new_order_array, nbr_of_elements))

        # extract order based on largest to smallest
        new_order_array = new_order_array[1:]
        new_values_for_scribble = np.argsort(new_order_array)[::-1]+1

        for s_idx in range(1,len(new_values_for_scribble)+1):
            Mask_w_scribble_id_ordered[Mask_w_scribble_id == new_values_for_scribble[s_idx-1]] = s_idx

        return Mask_w_scribble_id_ordered



# Test the module
if "__main__" == __name__ :
    # toy_example = np.array([[3, 3, 2, 0, 1, 0],
    #                         [3, 3, 3, 1, 0, 1],
    #                         [1, 0, 0, 1, 0, 2],
    #                         [0, 1, 1, 0, 2, 1],
    #                         [0, 0, 2, 2, 1, 2],
    #                         [0, 0, 2, 1, 1, 1]])

    toy_example = np.array([[1, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 1, 1, 1, 1, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 1, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

    # toy_example = (toy_example*2)-1
    # toy_example[0,0] = 0
    # print(toy_example)


    dfs = depth_first_search()
    Dict = dfs.extract_scribbles_properly(toy_example)
    print(Dict)