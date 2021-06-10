import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ..modules.consensus_network_modules import MutualMatching, NeighConsensus, FeatureCorrelation
from ..modules.mod import conv, predict_flow
from ..modules.feature_correlation_layer import FeatureL2Norm, GlobalFeatureCorrelationLayer
from ..modules.matching_modules import initialize_flow_decoder_, initialize_mapping_decoder_
from GOCor import local_gocor
from GOCor.optimizer_selection_functions import define_optimizer_local_corr
from GOCor.global_gocor_modules import GlobalGOCorWithFlexibleContextAwareInitializer


def pre_process_data_GLUNet(source_img, target_img, device, mean_vector=[0.485, 0.456, 0.406],
                            std_vector=[0.229, 0.224, 0.225], apply_flip=False):
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = target_img.shape

    # original resolution
    if h_scale < 256:
        int_preprocessed_height = 256
    else:
        int_preprocessed_height = int(math.floor(int(h_scale / 8.0) * 8.0))

    if w_scale < 256:
        int_preprocessed_width = 256
    else:
        int_preprocessed_width = int(math.floor(int(w_scale / 8.0) * 8.0))

    if apply_flip:
        # flip the target image horizontally
        target_img_original = target_img
        target_img = []
        for i in range(b):
            transformed_image = np.fliplr(target_img_original[i].cpu().permute(1, 2, 0).numpy())
            target_img.append(transformed_image)

        target_img = torch.from_numpy(np.uint8(target_img)).permute(0, 3, 1, 2)

    source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                      size=(int_preprocessed_height, int_preprocessed_width),
                                                      mode='area').byte()
    target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                      size=(int_preprocessed_height, int_preprocessed_width),
                                                      mode='area').byte()
    source_img_copy = source_img_copy.float().to(device).div(255.0)
    target_img_copy = target_img_copy.float().to(device).div(255.0)
    mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

    # resolution 256x256
    source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                      size=(256, 256),
                                                      mode='area').byte()
    target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                      size=(256, 256),
                                                      mode='area').byte()

    source_img_256 = source_img_256.float().div(255.0)
    target_img_256 = target_img_256.float().div(255.0)
    source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])

    ratio_x = float(w_scale) / float(int_preprocessed_width)
    ratio_y = float(h_scale) / float(int_preprocessed_height)
    return source_img_copy.to(device), target_img_copy.to(device), source_img_256.to(device), \
           target_img_256.to(device), ratio_x, ratio_y


class MatchingNetParams:
    """Class for network parameters."""
    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)

    def get(self, name: str, *default):
        """Get a parameter value with the given name. If it does not exists, it return the default value given as a
        second argument or returns an error if no default value is given."""
        if len(default) > 1:
            raise ValueError('Can only give one default value.')

        if not default:
            return getattr(self, name)

        return getattr(self, name, default[0])

    def has(self, name: str):
        """Check if there exist a parameter with the given name."""
        return hasattr(self, name)


class BaseMultiScaleMatchingNet(nn.Module):
    """
        Commun to all multiscale architectures
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = None

    @staticmethod
    def initialize_mapping_decoder(decoder_type, in_channels, batch_norm=True, **kwargs):
        return initialize_mapping_decoder_(decoder_type, in_channels, batch_norm, **kwargs)

    @staticmethod
    def initialize_flow_decoder(decoder_type, decoder_inputs,  in_channels_corr, nbr_upfeat_channels,
                                batch_norm=True, **kwargs):
        return initialize_flow_decoder_(decoder_type, decoder_inputs,  in_channels_corr, nbr_upfeat_channels,
                                        batch_norm, **kwargs)

    @staticmethod
    def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        if float(torch.__version__[:3]) >= 1.3:
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        else:
            output = nn.functional.grid_sample(x, vgrid)
        return output

    def set_epoch(self, epoch):
        self.epoch = epoch


class BaseGLUMultiScaleMatchingNet(BaseMultiScaleMatchingNet):
    """Base class for GLU-Net based networks."""

    def __init__(self, params, pyramid=None, pyramid_256=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params = params
        self.visdom = None
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU(inplace=False)

        if pyramid is not None:
            self.pyramid = pyramid
        if pyramid_256 is not None:
            self.pyramid_256 = pyramid_256

    def initialize_global_corr(self):
        if self.params.global_corr_type == 'GlobalGOCor':
            # Global GOCor with FlexibleContextAware Initializer module
            self.corr = GlobalGOCorWithFlexibleContextAwareInitializer(
                global_gocor_arguments=self.params.GOCor_global_arguments)
        elif self.params.global_corr_type == 'NC-Net':
            # Neighborhood consensus network
            ncons_kernel_sizes = [3, 3, 3]
            ncons_channels = [10, 10, 1]
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
            # normalisation is applied in code here
            self.NeighConsensus = NeighConsensus(use_cuda=True,
                                                 kernel_sizes=ncons_kernel_sizes,
                                                 channels=ncons_channels)
        else:
            # Feature correlation layer
            self.corr = GlobalFeatureCorrelationLayer(shape='3D', normalization=False,
                                                      put_W_first_in_channel_dimension=False)

    def initialize_local_corr(self):
        if self.params.local_corr_type == 'LocalGOCor':
            if self.params.same_local_corr_at_all_levels:
                # here same initializer and same optimizer applied at all levels
                initializer = local_gocor.LocalCorrSimpleInitializer()
                optimizer = define_optimizer_local_corr(self.params.GOCor_local_arguments)
                self.local_corr = local_gocor.LocalGOCor(filter_initializer=initializer, filter_optimizer=optimizer)
            else:
                initializer_3 = local_gocor.LocalCorrSimpleInitializer()
                optimizer_3 = define_optimizer_local_corr(self.params.GOCor_local_arguments)
                self.local_corr_3 = local_gocor.LocalGOCor(filter_initializer=initializer_3, filter_optimizer=optimizer_3)

                initializer_2 = local_gocor.LocalCorrSimpleInitializer()
                optimizer_2 = define_optimizer_local_corr(self.params.GOCor_local_arguments)
                self.local_corr_2 = local_gocor.LocalGOCor(filter_initializer=initializer_2, filter_optimizer=optimizer_2)

                initializer_1 = local_gocor.LocalCorrSimpleInitializer()
                optimizer_1 = define_optimizer_local_corr(self.params.GOCor_local_arguments)
                self.local_corr_1 = local_gocor.LocalGOCor(filter_initializer=initializer_1, filter_optimizer=optimizer_1)

    def get_global_correlation(self, c14, c24):
        b = c14.shape[0]
        if 'GOCor' in self.params.global_corr_type:
            if self.params.normalize_features:
                corr4, losses4 = self.corr(self.l2norm(c14), self.l2norm(c24))
            else:
                corr4, losses4 = self.corr(c14, c24)
        elif self.params.global_corr_type == 'NC-Net':
            if self.params.normalize_features:
                corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            else:
                corr4d = self.corr(c24, c14)  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4d = self.NeighConsensus(corr4d)
            corr4d = MutualMatching(corr4d)  # size is [b, 1, hsource, wsource, htarget, wtarget]
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        else:
            # directly obtain the 3D correlation
            if self.params.normalize_features:
                corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            else:
                corr4 = self.corr(c24, c14)  # first source, then target

        if self.params.cyclic_consistency:
            # to add on top of the correlation ! (already included in NC-Net)
            corr4d = MutualMatching(corr4.view(b, c24.shape[2], c24.shape[3], c14.shape[2], c14.shape[3]).unsqueeze(1))
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])

        if self.params.normalize == 'l2norm':
            corr4 = self.l2norm(corr4)
        elif self.params.normalize == 'relu_l2norm':
            corr4 = self.l2norm(F.relu(corr4))
        elif self.params.normalize == 'leakyrelu':
            corr4 = self.leakyRELU(corr4)
        return corr4

    def initialize_last_level_refinement_module(self, input_to_refinement, batch_norm):
        self.l_dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                               batch_norm=batch_norm)
        self.l_dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.l_dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.l_dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.l_dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.l_dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.l_dc_conv7 = predict_flow(32)

    def initialize_adaptive_reso_refinement_module(self,  input_to_refinement, batch_norm):
        self.dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                             batch_norm=batch_norm)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

    def PWCNetRefinementAdaptiveReso(self, x):
        x = self.dc_conv6(self.dc_conv5(self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))))
        res = self.dc_conv7(x)
        return x, res

    def PWCNetRefinementFinal(self, x):
        x = self.l_dc_conv6(self.l_dc_conv5(self.l_dc_conv4(self.l_dc_conv3(self.l_dc_conv2(self.l_dc_conv1(x))))))
        res = self.l_dc_conv7(x)
        return x, res

    def extract_features(self, im_target, im_source, im_target_256, im_source_256,
                         im1_pyr=None, im2_pyr=None, im1_pyr_256=None, im2_pyr_256=None):
        # pyramid, original reso
        if im1_pyr is None:
            im1_pyr = self.pyramid(im_target, eigth_resolution=True)
        if im2_pyr is None:
            im2_pyr = self.pyramid(im_source, eigth_resolution=True)
        c11 = im1_pyr[-2]  # size original_res/4xoriginal_res/4
        c21 = im2_pyr[-2]
        c12 = im1_pyr[-1]  # size original_res/8xoriginal_res/8
        c22 = im2_pyr[-1]

        # pyramid, 256 reso
        if im1_pyr_256 is None:
            im1_pyr_256 = self.pyramid(im_target_256)
            im2_pyr_256 = self.pyramid(im_source_256)
        c13 = im1_pyr_256[-4]
        c23 = im2_pyr_256[-4]
        c14 = im1_pyr_256[-3]
        c24 = im2_pyr_256[-3]
        return c14, c24, c13, c23, c12, c22, c11, c21

    def pre_process_data(self, im_source, im_target, apply_flip=False):
        return pre_process_data_GLUNet(im_source, im_target, self.params.device, apply_flip=apply_flip)

    def estimate_flow_coarse_reso(self, source_img, target_img, device):

        source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y \
            = self.pre_process_data(source_img, target_img, device)
        output_256, output = self.forward(target_img, source_img, target_img_256, source_img_256)

        flow_est_list = output_256['flow_estimates']
        flow_est = flow_est_list[0]  # coarsest level
        return flow_est

    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first'):
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale*scaling), int(w_scale*scaling))

        source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y \
            = self.pre_process_data(source_img, target_img)
        output_256, output = self.forward(target_img, source_img, target_img_256, source_img_256)

        flow_est_list = output['flow_estimates']
        flow_est = flow_est_list[-1]

        if output_shape is not None:
            ratio_x *= float(output_shape[1]) / float(w_scale)
            ratio_y *= float(output_shape[0]) / float(h_scale)
        else:
            output_shape = (h_scale, w_scale)
        flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape, mode='bilinear',
                                                   align_corners=False)

        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y

        if mode == 'channel_first':
            return flow_est
        else:
            return flow_est.permute(0, 2, 3, 1)


def set_glunet_parameters(global_corr_type='feature_corr_layer', gocor_global_arguments=None, normalize='relu_l2norm',
                          normalize_features=True, cyclic_consistency=False, md=4,
                          local_corr_type='feature_corr_layer', gocor_local_arguments=None,
                          same_local_corr_at_all_levels=True, local_decoder_type='OpticalFlowEstimator',
                          global_decoder_type='CMDTop', decoder_inputs='corr_flow_feat',
                          refinement_at_adaptive_reso=True, refinement_at_all_levels=False,
                          refinement_at_finest_level=True, apply_refinement_finest_resolution=True,
                          give_flow_to_refinement_module=False, batch_norm=True, nbr_upfeat_channels=2):
    params = MatchingNetParams()
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.GOCor_global_arguments = gocor_global_arguments
    params.global_corr_type = global_corr_type
    params.normalize = normalize
    params.normalize_features = normalize_features
    params.cyclic_consistency = cyclic_consistency

    params.local_corr_type = local_corr_type
    params.GOCor_local_arguments = gocor_local_arguments
    params.same_local_corr_at_all_levels = same_local_corr_at_all_levels

    params.local_decoder_type = local_decoder_type
    params.global_decoder_type = global_decoder_type
    params.decoder_inputs = decoder_inputs

    params.refinement_at_adaptive_reso = refinement_at_adaptive_reso
    params.refinement_at_all_levels = refinement_at_all_levels
    params.refinement_at_finest_level = refinement_at_finest_level
    params.apply_refinement_finest_resolution = apply_refinement_finest_resolution
    params.give_flow_to_refinement_module = give_flow_to_refinement_module

    params.batch_norm = batch_norm
    params.md = md
    params.nbr_upfeat_channels = nbr_upfeat_channels
    return params
