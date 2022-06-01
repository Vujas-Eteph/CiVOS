# Image operations, drawing, scalling, saving, rading,...
from click_davisinteractive import utils as interactive_utils
from click_davisinteractive.utils.visualization import _pascal_color_map

import numpy as np
import glob
import os
import cv2
from PIL import Image, ImageDraw


# Select files paths accordingly to the PC's name
path_2_DAVIS_scribbles = "../DAVIS/2017/trainval/JPEGImages/480p/"


class PIL_DAVIS_image_operations_class():
    def __init__(self):
        self.cmap = _pascal_color_map(normalized=False)

    def __len__(self):
        return len(self.__all_frames)

    def get_loaded_sequence(self):
        return self.__all_frames

    def load_sequence_images(self, sequence_name):
        self.sequence_name = sequence_name
        self.__all_frames  = sorted(glob.glob(path_2_DAVIS_scribbles + self.sequence_name + "/*"))

    def read_frame_anno(self, frame_anno):
        self.frame_anno = frame_anno
        self.frame = Image.open(self.__all_frames[frame_anno])

    def draw_scribbles_on_frame(self, scribble, frame_anno):
        shape      = np.array(self.frame).shape[:2][::-1]
        self.frame = interactive_utils.visualization.draw_scribble(self.frame, scribble, frame_anno, output_size=shape, width=3)

    def return_frame(self):
        return self.frame

    def draw_points_on_frame(self, clicks_d, radius = 3):
        drawng_methds = ImageDraw.Draw(self.frame, mode=None)
        shape = np.array(self.frame).shape[:2][::-1]
        keys = clicks_d.keys()
        for key in keys:
            obj_id = int(key.split("_")[-1])
            obj_points = clicks_d[key]
            click_rescaled_to_image = (obj_points * shape).astype(np.int32)
            points_for_PIL = []
            for idx in range(-radius,radius+1):
                for jdx in range(-radius, radius + 1):
                    points_for_PIL.append(tuple(click_rescaled_to_image + [idx, jdx]))

            drawng_methds.point((points_for_PIL), fill=tuple(self.cmap[obj_id] + [100, 100, 100]))

    def draw_points_on_frame_multiple(self, clicks_d, radius = 3):
        drawng_methds = ImageDraw.Draw(self.frame, mode=None)
        shape = np.array(self.frame).shape[:2][::-1]
        keys = clicks_d.keys()
        for key in keys:
            obj_id = int(key.split("_")[-1])
            obj_points = clicks_d[key]
            for elem in obj_points:
                click_rescaled_to_image = (elem[0][::-1] * shape).astype(np.int32)
                points_for_PIL = []
                for idx in range(-radius,radius+1):
                    for jdx in range(-radius, radius + 1):
                        points_for_PIL.append(tuple(click_rescaled_to_image + [idx, jdx]))

                drawng_methds.point((points_for_PIL), fill=tuple(self.cmap[obj_id] + [100, 100, 100]))


    def draw_points_on_frame(self, points_for_scribbles, radius = 3):
        drawng_methds = ImageDraw.Draw(self.frame, mode=None)
        shape = np.array(self.frame).shape[:2][::-1]

        points_2_draw = points_for_scribbles['scribbles'][self.frame_anno]

        for point_2_draw in points_2_draw:
            obj_id = point_2_draw['object_id']
            obj_points = np.array(point_2_draw['path'][0])
            point_2_draw   = (obj_points*shape).astype(np.int32)
            points_for_PIL = []
            for idx in range(-radius,radius+1):
                for jdx in range(-radius, radius + 1):
                    points_for_PIL.append(tuple(point_2_draw + [idx, jdx]))

            # print(self.cmap[obj_id])
            drawng_methds.point((points_for_PIL), fill=tuple(self.cmap[obj_id] + [100, 100, 100]))
            drawng_methds.point(points_for_PIL[int((2*radius+1)*(2*radius+1)/2)], fill=tuple(self.cmap[obj_id]))


    def draw_points_on_frame_bis(self, inputs, radius = 3):
        drawng_methds = ImageDraw.Draw(self.frame, mode=None)
        shape = np.array(self.frame).shape[:2][::-1]

        for obj_id, point_2_draw in enumerate(inputs):
            obj_points = point_2_draw
            point_2_draw   = (obj_points*shape).astype(np.int32)
            points_for_PIL = []
            for idx in range(-radius,radius+1):
                for jdx in range(-radius, radius + 1):
                    points_for_PIL.append(tuple(point_2_draw + [idx, jdx]))

            # print(self.cmap[obj_id])
            drawng_methds.point((points_for_PIL), fill=tuple(self.cmap[obj_id] + [0, 0, 0]))
            drawng_methds.point(points_for_PIL[int((2*radius+1)*(2*radius+1)/2)], fill=tuple(self.cmap[obj_id]+ [100, 100, 100]))

    def Scribble_2_Mask_Visu(self, scribbles):
        # Scribble_2_mask
        shape      = np.array(self.frame).shape[:2]
        self.Empty_mask = np.ones([shape[0],shape[1],3])*255

        points_2_draw = scribbles['scribbles'][self.frame_anno]
        for elem in points_2_draw:
            obj_id          = elem['object_id']
            # print("obj_id",obj_id)
            obj_points      = np.array(elem['path'])
            points_for_mask = (obj_points[:,::-1]*shape).astype(np.int32)

            self.Empty_mask[points_for_mask[:,0],points_for_mask[:,1],:] = self.cmap[obj_id][::-1]

    def Rectangle(self, scribbles):
        # Group the scribbles types together
        points_2_draw_assemble = scribbles['scribbles'][self.frame_anno]
        points_2_assemble = np.empty((0,3))
        for elem_2_assemble in points_2_draw_assemble:
            obj_id = elem_2_assemble['object_id']
            obj_points = np.array(elem_2_assemble['path'])
            obj_id_vec = np.ones((obj_points.shape[0],1))*obj_id
            compose_points_with_id = np.concatenate((obj_id_vec, obj_points), axis = 1)
            points_2_assemble = np.append(points_2_assemble, compose_points_with_id, axis = 0)

        assembled_points_d = {}
        for obj_id in np.unique(points_2_assemble[:,0]).astype(np.int):
            key_n = "Obj_id_{}".format(obj_id)
            points_to_assemble = points_2_assemble[obj_id == points_2_assemble[:,0],:]
            assembled_points_d[key_n] = points_to_assemble[:,1::]

        points_for_scribbles = []
        mean_points = []

        shape = np.array(self.frame).shape[:2]
        # points_2_draw = scribbles['scribbles'][self.frame_anno]
        for key in assembled_points_d.keys():
            obj_id     = int(key.split("_")[-1])
            obj_points = assembled_points_d[key]

            # Check if there are duplicated
            _, counts = np.unique(obj_points, return_counts=True, axis = 1)
            # print("COUNTS", counts)

            # Draw central point
            yx_mean = np.mean(obj_points,axis=0)
            mean_position = (yx_mean[::-1] * shape).astype(np.int32)
            self.Empty_mask[mean_position[0], mean_position[1],:] = [0,0,0]

            radius = 1
            for idx in range(-radius,radius+1):
                for jdx in range(-radius, radius + 1):
                    self.Empty_mask[mean_position[0]+idx, mean_position[1]+jdx, :] = [0, 0, 0]


            # Find shortes distante point and draw it
            dist_yx = (yx_mean - obj_points)**2
            dist_yx = np.sum(dist_yx, axis = 1)
            closest_points = dist_yx.argmin()

            points_for_scribbles.append(obj_points[closest_points])
            mean_points.append(yx_mean)

            closest_2_mean_point = (obj_points[closest_points,::-1] * shape).astype(np.int32)
            self.Empty_mask[closest_2_mean_point[0], closest_2_mean_point[1], :] = [250, 250, 0]

            radius = 1
            for idx in range(-radius,radius+1):
                for jdx in range(-radius, radius + 1):
                    y = closest_2_mean_point[0] + idx
                    x = closest_2_mean_point[1] + jdx
                    y = 0 if y < 0 else y
                    x = 0 if x < 0 else x
                    y = self.Empty_mask.shape[0]-1 if y >= self.Empty_mask.shape[0] else y
                    x = self.Empty_mask.shape[1]-1 if x >= self.Empty_mask.shape[1] else x
                    try:
                        self.Empty_mask[y, x, :] = [250, 250, 0]
                    except IndexError:
                        print('hi')

            obj_points = (obj_points[:,::-1]*shape).astype(np.int32)
            h_low  = obj_points[:,0].min()
            h_high = obj_points[:,0].max()
            w_left  = obj_points[:,1].min()
            w_right = obj_points[:,1].max()
            length_h = h_high - h_low
            length_w = w_right - w_left

            line_h_low  = np.ones([length_w,2]).astype(np.int32)
            line_h_high = np.ones([length_w,2]).astype(np.int32)
            line_h_low[:,0],line_h_low[:,1]   = line_h_low[:,0]*h_low,line_h_low[:,1]*np.arange(w_left, w_right)
            line_h_high[:,0],line_h_high[:,1] = line_h_high[:,0]*h_high,line_h_high[:,1]*np.arange(w_left, w_right)

            line_w_left  = np.ones([length_h,2]).astype(np.int32)
            line_w_right = np.ones([length_h,2]).astype(np.int32)
            line_w_left[:,0], line_w_left[:,1]   = line_w_left[:,0]*np.arange(h_low, h_high), line_w_left[:,1]*w_left
            line_w_right[:,0], line_w_right[:,1] = line_w_right[:,0]*np.arange(h_low, h_high), line_w_right[:,1]*w_right

            # Draw rectangle
            self.Empty_mask[line_h_low[:, 0], line_h_low[:, 1], :] = self.cmap[obj_id][::-1]
            self.Empty_mask[line_h_high[:, 0], line_h_high[:, 1], :] = self.cmap[obj_id][::-1]
            self.Empty_mask[line_w_left[:, 0], line_w_left[:, 1], :] = self.cmap[obj_id][::-1]
            self.Empty_mask[line_w_right[:, 0], line_w_right[:, 1], :] = self.cmap[obj_id][::-1]


        print("points_for_scribbles", points_for_scribbles)
        print("mean_points", mean_points)

    def new_center(self, scribbles):
        shape = np.array(self.frame).shape[:2]
        points_2_draw = scribbles['scribbles'][self.frame_anno]
        for elem in points_2_draw:
            obj_id = elem['object_id']
            obj_points = np.array(elem['path'])
            obj_points = (obj_points[:, ::-1] * shape).astype(np.int32)
            pass


    def Show_scribbles_raw(self, name_w = "Mask_of_scribbles"):
        cv2.imshow(name_w, self.Empty_mask)

        # cv2.destroyAllWindows()#

    def Add_mean_click_and_click_i_want(self, input1, input2):
        shape = np.array(self.frame).shape[:2]
        radius = 1

        # Mean
        for obj_id, point_2_draw in enumerate(input1):
            obj_points   = point_2_draw
            point_2_draw = (obj_points[::-1]*shape).astype(np.int32)
            for idx in range(-radius,radius+1):
                for jdx in range(-radius, radius + 1):
                    self.Empty_mask[point_2_draw[0] + idx, point_2_draw[1] + jdx, :] = self.cmap[obj_id]

        # Click
        for elem in (input2['scribbles'][self.frame_anno]):
            obj_points = np.array(elem['path'][0])
            obj_id = elem['object_id']
            point_2_draw = (obj_points[::-1]*shape).astype(np.int32)
            for idx in range(-radius,radius+1):
                for jdx in range(-radius, radius + 1):
                    self.Empty_mask[point_2_draw[0] + idx, point_2_draw[1] + jdx, :] = self.cmap[obj_id][::-1]


    def show_image(self):
        image_for_opencv = np.array(self.frame)[:,:,::-1]
        while True:
            cv2.imshow(self.sequence_name, image_for_opencv)
            key = cv2.waitKey(0)

            if ord('q')==key:
                break

        cv2.destroyAllWindows()

    def save_image(self, c_round, frame_nbr, path_2_save_2=None):
        if path_2_save_2 is None:
            path_0 = './images_zzz'
            path_2_save_2 = os.path.join(path_0, self.sequence_name)
            if not os.path.exists(path_2_save_2):
                os.mkdir(path_2_save_2)

        path_for_image = os.path.join(path_2_save_2, 'round_{0}_frame_nbr_{1:05d}.png'.format(c_round, frame_nbr))

        self.frame.save(path_for_image, "png")

        

