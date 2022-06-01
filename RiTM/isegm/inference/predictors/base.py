import torch
import torch.nn.functional as F
from torchvision import transforms
from RiTM.isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())


    def set_input_image(self, image):
        image_nd = self.to_tensor(image)    # transforms.ToTensor() Convert a numpy.drarray or PIL image to a tensor
        for transform in self.transforms:
            # reset all values of transformation and image, so that we get back to the original image
            transform.reset()   # i.e., RiTM.isegm.inference.transforms.zoom_in.ZoomIn functions

        self.original_image = image_nd.to(self.device)  # put tensor of the image on a designated device, here a GPU

        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)                  # Add a dimension for the batch []
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])   # Set previsous prediction ot zeros, and only take one channel


    def set_input_image_but_keep_transformations_vujas(self, image):
        image_nd = self.to_tensor(image)    # transforms.ToTensor() Convert a numpy.drarray or PIL image to a tensor

        self.original_image = image_nd.to(self.device)  # put tensor of the image on a designated device, here a GPU

        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0) # Add a dimension for the batch []

        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])


    def get_prediction(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()  # Get full click list
        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        # Set up previous mask for the first time, if none where fiven before
        if prev_mask is None:
            prev_mask = self.prev_prediction
        # But his seems to be the classical way to incorporate Encoded Clicks in the Network, it can either be DMF or Conv1E (see Fig 3 RiTM paper)
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        # Apply a transformation
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(input_image, [clicks_list])

        # Prediction
        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)

        # Rescale prediction the input to image size using bilinear interopolation
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = prediction # Save new prediction for latter usage
        
        return prediction.cpu().numpy()[0, 0]  # Return as a numpy array

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):    #-> OG: self.get_prediction
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)['instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        # Check if there is any transformations
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)   # Transform the image with the clicks
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):  #-> OG: self._get_prediction
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
