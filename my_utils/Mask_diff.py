# Simulate de First Click based on the original paper FCANet
#
# by Stéphane Vujasinovic

import numpy as np



def mask_diff(y_GT, y_pred):
    """
    y_GT -> Gt mask
    y_pred -> prediction
    :param y_GT:
    :param y_pred:
    :return:
    """
    Complete_False_negative_mask = np.zeros((y_GT.max(),y_GT.shape[0],y_GT.shape[1]))
    Complete_False_positive_mask = np.zeros((y_GT.max(),y_GT.shape[0],y_GT.shape[1]))

    for obj_id in range(1, y_GT.max()+1):
        GT_obj_id   = (y_GT==obj_id).astype(np.int)
        Pred_obj_id = (y_pred==obj_id).astype(np.int)
        mask_diff = GT_obj_id - Pred_obj_id
        False_negative_mask = (mask_diff==1).astype(np.int)     # Where positive clicks have to be inferred
        False_positive_mask = (mask_diff==-1).astype(np.int)    # WHere negative clicks have to be inferred

        Complete_False_negative_mask[obj_id-1] = False_negative_mask
        Complete_False_positive_mask[obj_id-1] = False_positive_mask


    return Complete_False_negative_mask.astype(np.int), Complete_False_positive_mask.astype(np.int)



# Test the module
if "__main__" == __name__:
    toy_ex_pred = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
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

    toy_ex_GT = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,1,1,0,0,0,0,0,0,0,3,3,3,0],
                          [0,0,1,1,1,1,0,0,0,0,0,0,0,3,0,0],
                          [0,0,0,0,0,1,0,0,0,0,0,0,0,3,0,0],
                          [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                          [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
                          [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0],
                          [0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0],
                          [0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0],
                          [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                          [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,2,0,2,0,0,0,0,0,3,3,3,0,3,0,0],
                          [0,0,0,0,0,0,0,0,0,3,0,0,0,3,0,0],
                          [0,2,0,2,0,0,0,0,0,3,3,3,3,3,0,0]])

    FN, FP = mask_diff(toy_ex_GT, toy_ex_pred)
    print('hi')

