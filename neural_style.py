import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.optim import optimizer
from torchvision import models
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm
from datetime import datetime as dt

import torch
import comfy.utils
import pickle
import math

CONTENT_WEIGHT = 8  # "alpha" in the literature (default: 8)
STYLE_WEIGHT = 70  # "beta" in the literature (default: 70)
TV_WEIGHT = 10 # (default: 10)
FINAL_LOSS = 0.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STYLE_LAYERS_DEFAULT = {
    'relu1_1': 1.0,
    'relu2_1': 1.0,
    'relu3_1': 1.0,
    'relu4_1': 1.0,
    'relu5_1': 1.0,
}

CONTENT_LAYERS_DEFAULT = ('relu4_2', )

class NormalizeContentWithAnalyticGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = (grad_output / (grad_output.abs().sum() + 1e-8))
        global CONTENT_WEIGHT
        grad_x_ = CONTENT_WEIGHT * grad_x
        return grad_x_

class NormalizeContentModule(nn.Module):
    def forward(self, x):
        return NormalizeContentWithAnalyticGrad.apply(x)

class NormalizeStyleWithAnalyticGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = (grad_output / (grad_output.abs().sum() + 1e-8))
        global STYLE_WEIGHT
        grad_x_ = STYLE_WEIGHT * grad_x
        return grad_x_

class NormalizeStyleModule(nn.Module):
    def forward(self, x):
        return NormalizeStyleWithAnalyticGrad.apply(x)

class TVLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (0.0*x.sum())
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        B, C, H, W = x.shape
        grad_input = torch.zeros_like(x)
        x_diff = x[:, :, 0:H-1, 0:W-1] - x[:, :, 0:H-1, 1:W]
        y_diff = x[:, :, 0:H-1, 0:W-1] - x[:, :, 1:H,   0:W-1]
        grad_input[:, :, 0:H-1, 0:W-1] += x_diff
        grad_input[:, :, 0:H-1, 0:W-1] += y_diff
        grad_input[:, :, 0:H-1, 1:W]   += -x_diff
        grad_input[:, :, 1:H,   0:W-1] += -y_diff
        global TV_WEIGHT
        grad_input = grad_input * TV_WEIGHT
        return grad_input

class TVLossModule(nn.Module):
    def forward(self, x):
        return TVLoss.apply(x)

def content_loss_func(target_features, content_features, norm):
    content_loss = 0.0
    for layer in content_features:
        target_feature = target_features[layer]
        if norm:
            m = NormalizeContentModule()
            target_feature_ = m(target_feature)
        else:
            target_feature_ = target_feature
        content_feature = content_features[layer]
        content_layer_loss = F.mse_loss(target_feature_, content_feature)
        content_loss += content_layer_loss
    return content_loss

def style_loss_func(target_features, style_features, precomputed_style_grams, norm):
    style_loss = 0.0
    for layer in style_features:
        target_feature = target_features[layer]
        if norm:
            m = NormalizeStyleModule()
            target_feature_ = m(target_feature)
        else:
            target_feature_ = target_feature
        target_gram = gram_matrix(target_feature_,True)
        style_gram = precomputed_style_grams[layer]
        layer_style_loss = STYLE_LAYERS_DEFAULT[layer] * F.mse_loss(target_gram, style_gram)
        style_loss += layer_style_loss
    return style_loss

def get_features(image: Tensor, model:nn.Module, layers=None):
    if layers is None:
        layers = tuple(STYLE_LAYERS_DEFAULT) + CONTENT_LAYERS_DEFAULT
    features = {}
    block_num = 1
    conv_num = 0
    relu_num = 0
    x = image
    for layer in model:
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            # produce layer name to find matching convolutions from the paper
            # and store their output for further processing.
            conv_num += 1
            name = f'conv{block_num}_{conv_num}'
            if name in layers:
                features[name] = x
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            # In VGG, each block ends with max/avg pooling layer.
            block_num += 1
            conv_num = 0
            relu_num = 0
        elif isinstance(layer, nn.ReLU):
            relu_num += 1
            name = f'relu{block_num}_{relu_num}'
            if name in layers:
                features[name] = x
        elif isinstance(layer, nn.BatchNorm2d):
            pass
        else:
            raise Exception(f'Unknown layer: {layer}')
    return features

def gram_matrix(input: Tensor, normalize=False) -> Tensor:
    (b, ch, h, w) = input.size()
    # resise F_XL into \hat F_XL
    features = input.view(b * ch, h * w)
    # compute the gram product
    gram = torch.mm(features, features.t())
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    if normalize:
        gram /= input.nelement()  # equivalent to: gram = gram.div(b * ch * h * w)
    return gram

class NeuralStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content": ("IMAGE", ),
                "style1": ("IMAGE", ),
                "content_weight": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.00,
                    "round": 0.001,
                    "display": "number",
                    "lazy": True
                }),
                "style1_weight": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.00,
                    "round": 0.001,
                    "display": "number",
                }),
                "style2_weight": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.00,
                    "round": 0.001,
                    "display": "number",
                }),
                "style3_weight": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.00,
                    "round": 0.001,
                    "display": "number",
                }),
                "total_variation_weight": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.0001,
                    "round": 0.0001,
                    "display": "number",
                }),
                "iterations": ("INT", {
                    "default": 500,
                    "min": 0.0,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number",
                }),
                "normalize_gradients": (["disabled", "enabled"],),
                "force_grayscale": (["no", "yes"],),
                "optimizer": (["L-BFGS", "ADAM"],),
                "tile_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 1,
                    "display": "number",
                }),
                "tile_overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "style2": ("IMAGE", ),
                "style3": ("IMAGE", ),
                "init_image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE","FLOAT",)
    RETURN_NAMES = ("result","final_loss",)
    FUNCTION = "batched_operation"

    CATEGORY = "image"

    def img_transform(self, x, output_size, exact_size = None):
        if exact_size is not None:
            x = T.Resize(exact_size)(x)
        x = x[:,[2,1,0],:,:]
        B,C,H,W = x.shape
        if W > H:
            #NB: you'd expect round() instead of int(), but i want to match Lua's image.scale(), because that is used in https://github.com/jcjohnson/neural-style/
            h = int( ( output_size * H ) / W )
            w = output_size
        else:
            h = output_size
            w = int( ( output_size * W ) / H )
        transform = T.Compose([
            # Smaller edge of the image will be matched to `img_size`
            T.Resize((h,w)),
        ])
        x = transform(x)
        x = x * 256.0
        mean_pixel = torch.tensor([103.939, 116.779, 123.68]).to('cuda')
        mean_pixel = mean_pixel.view(1, 3, 1, 1).expand(x.size(0), 3, x.size(2), x.size(3))
        x = x - mean_pixel
        return x

    def inv_img_transform(self, t) -> Tensor:
        mean_pixel = torch.tensor([103.939, 116.779, 123.68])
        mean_pixel = mean_pixel.view(1, 3, 1, 1).expand(t.size(0), t.size(1), t.size(2), t.size(3))
        result = t + mean_pixel
        result = result / 256.0
        result = result[:,[2,1,0],:,:]
        return result

    def replace_maxpool_ceil(self,module):
        for name, child in list(module.named_children()):
            # If child is a MaxPool2d, replace with same params but ceil_mode=True
            if isinstance(child, nn.MaxPool2d):
                new = nn.MaxPool2d(
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    return_indices=child.return_indices,
                    ceil_mode=True
                )
                setattr(module, name, new)
            else:
                # recurse into children (works for Sequential, ModuleList, etc.)
                self.replace_maxpool_ceil(child)

    # farted out by ChatGPT-5.4 mini
    def linear_weight_mask(self, tile_h, tile_w, overlap_h, overlap_w, device=None, dtype=torch.float32):
        y = torch.ones(tile_h, device=device, dtype=dtype)
        x = torch.ones(tile_w, device=device, dtype=dtype)
        if overlap_h > 0:
            ramp = torch.linspace(0, 1, overlap_h + 2, device=device, dtype=dtype)[1:-1]
            y[:overlap_h] = ramp
            y[-overlap_h:] = ramp.flip(0)
        if overlap_w > 0:
            ramp = torch.linspace(0, 1, overlap_w + 2, device=device, dtype=dtype)[1:-1]
            x[:overlap_w] = ramp
            x[-overlap_w:] = ramp.flip(0)
        return torch.outer(y, x)

    # ditto, but slightly modified
    def tile_process_blend_b1_hwc(self, content_img, init_img, process_fn, tile_size=(512, 512), overlap=(64, 64)):
        """
        content_img: Tensor [1, H, W, C]
        process_fn: function that takes [1, th, tw, C] or [th, tw, C] and returns same shape
        """
        assert content_img.dim() == 4 and content_img.shape[0] == 1, "Expected [1, H, W, C]"
        _, H, W, C = content_img.shape
        th, tw = tile_size
        oh, ow = overlap
        sh, sw = th - oh, tw - ow
        device, dtype = content_img.device, content_img.dtype
        out = torch.zeros((1, H, W, C), device=device, dtype=torch.float32)
        weight_sum = torch.zeros((1, H, W, 1), device=device, dtype=torch.float32)
        wmask = self.linear_weight_mask(th, tw, oh, ow, device=device, dtype=torch.float32)[None, :, :, None]
        ys = list(range(0, max(H - th, 0) + 1, sh))
        xs = list(range(0, max(W - tw, 0) + 1, sw))
        if ys[-1] != max(H - th, 0):
            ys.append(max(H - th, 0))
        if xs[-1] != max(W - tw, 0):
            xs.append(max(W - tw, 0))
        for y in ys:
            for x in xs:
                content_tile = content_img[:, y:y+th, x:x+tw, :]
                init_img_tile = init_img[:, y:y+th, x:x+tw, :]
                pad_h = th - content_tile.shape[1]
                pad_w = tw - content_tile.shape[2]
                if pad_h > 0 or pad_w > 0:
                    content_tile = F.pad(content_tile, (0, 0, 0, pad_w, 0, pad_h))
                    init_img_tile = F.pad(init_img_tile, (0, 0, 0, pad_w, 0, pad_h))
                output_tile = process_fn((content_tile,init_img_tile))
                output_tile = output_tile[:, :min(th, H - y), :min(tw, W - x), :]
                wm = wmask[:, :output_tile.shape[1], :output_tile.shape[2], :]
                out[:, y:y+output_tile.shape[1], x:x+output_tile.shape[2], :] += output_tile.to(torch.float32) * wm
                weight_sum[:, y:y+output_tile.shape[1], x:x+output_tile.shape[2], :] += wm
        out = out / weight_sum.clamp_min(1e-8)
        return out.to(dtype)

    def tiled_operation(self, content, style1, content_weight, style1_weight, style2_weight, style3_weight, total_variation_weight, iterations, seed, normalize_gradients, force_grayscale, tile_size, tile_overlap, optimizer, style2: Tensor=None, style3: Tensor=None, init_image: Tensor=None):
        tile_size_ = tile_size
        tile_overlap_ = tile_overlap
        #when operating on tiles, the style image must be resized to the size of the original content, not the size of the processed tile. otherwise the style scale will be wrong.
        style_size = max(content.size(1),content.size(2))
        if init_image is None:
            init_image = torch.rand_like(content)
        out = self.tile_process_blend_b1_hwc(content, init_image, process_fn=lambda x:((self.untiled_operation(x[0], style1, content_weight, style1_weight, style2_weight, style3_weight, total_variation_weight, iterations, seed, normalize_gradients, force_grayscale, optimizer, style_size, style2, style3, x[1]))[0]), tile_size=(tile_size_, tile_size_), overlap=(tile_overlap_, tile_overlap_))
        return (out,0.0,)

    # neural_style() can not handle batch size > 1, this method can. this method does not operate on a batch in parallel, as is usual, because that would be pointless:
    # the VRAM requirement would be much higher and almost all GPUs will already be close to 100% utilization with just 1 image.
    def batched_operation(self, content, style1, content_weight, style1_weight, style2_weight, style3_weight, total_variation_weight, iterations, seed, normalize_gradients, force_grayscale, tile_size, tile_overlap, optimizer, style2: Tensor=None, style3: Tensor=None, init_image: Tensor=None):
        targets = []
        final_losses = []
        range_ = max(content.shape[0],style1.shape[0])
        if style2 is not None:
            range_ = max(range_,style2.shape[0])
        if style3 is not None:
            range_ = max(range_,style3.shape[0])
        if init_image is not None:
            range_ = max(range_,init_image.shape[0])
        for i in range(range_):
            content_index = i % ( content.shape[0] )
            content_ = content[content_index:content_index+1]
            style1_index = i % ( style1.shape[0] )
            style1_ = style1[style1_index:style1_index+1]
            if style2 is None:
                style2_ = None
            else:
                style2_index = i % ( style2.shape[0] )
                style2_ = style2[style2_index:style2_index+1]
            if style3 is None:
                style3_ = None
            else:
                style3_index = i % ( style3.shape[0] )
                style3_ = style3[style3_index:style3_index+1]
            if init_image is None:
                init_image_ = None
            else:
                init_image_index = i % ( init_image.shape[0] )
                init_image_ = init_image[init_image_index:init_image_index+1]
            if 0 == tile_size:
                target_,final_loss_ = self.untiled_operation(content_, style1_, content_weight, style1_weight, style2_weight, style3_weight, total_variation_weight, iterations, seed, normalize_gradients, force_grayscale, optimizer, None, style2_, style3_, init_image_)
            else:
                target_,final_loss_ = self.tiled_operation(content_, style1_, content_weight, style1_weight, style2_weight, style3_weight, total_variation_weight, iterations, seed, normalize_gradients, force_grayscale, tile_size, tile_overlap, optimizer, style2_, style3_, init_image_)
            targets.append(target_)
            final_losses.append(torch.tensor([final_loss_]))
        return (torch.cat(targets,dim=0),torch.cat(final_losses,dim=0),)

    def untiled_operation(self, content, style1, content_weight, style1_weight, style2_weight, style3_weight, total_variation_weight, iterations, seed, normalize_gradients, force_grayscale, optimizer, style_size=None, style2: Tensor=None, style3: Tensor=None, init_image: Tensor=None):
        with torch.torch.inference_mode(False):
                optimizer_option = "adam"
                learning_rate_option = 10.0
                np.random.seed(int(seed))
                torch.manual_seed(int(seed))
                dtype = torch.float32
                device = 'cuda'
                torch.backends.cudnn.benchmark = True
                if style2 is None:
                    style2_weight = 0.0
                if style3 is None:
                    style3_weight = 0.0
                # We will use frozen pretrained VGG neural network for feature extraction
                # In the original paper, authors have used VGG19 (without bn)
                model = models.vgg19(weights=None)
                #model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
                path = 'vgg19-d01eb7cb.pth'
                model.load_state_dict(torch.load(path))
                model = model.features
                self.replace_maxpool_ceil(model)
                # Authors in the original paper suggested use of AvgPool instead of MaxPool for more pleasing results.
                # However changing the pooling also affects activation, so the input needs to be scaled (not implemented).
                #for i, layer in enumerate(model):
                #    if isinstance(layer, torch.nn.MaxPool2d):
                #        model[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
                model = model.eval().requires_grad_(False).to(dtype).to(device)
                output_size = max(content.size(1),content.size(2))
                # The "content" image on which we apply style
                content_ = content.to(dtype).to(device)
                content_ = self.img_transform(torch.transpose(content_,1,3).squeeze().unsqueeze(0),output_size)
                if style_size is None:
                    style_size = output_size
                # The "style" image from which we obtain style
                style = style1.to(dtype).to(device)
                style = self.img_transform(torch.transpose(style,1,3).squeeze().unsqueeze(0),style_size)
                if style2 is None:
                    style_2 = None
                else:
                    style_2 = style2.to(dtype).to(device)
                    style_2 = self.img_transform(torch.transpose(style_2,1,3).squeeze().unsqueeze(0),style_size)
                if style3 is None:
                    style_3 = None
                else:
                    style_3 = style2.to(dtype).to(device)
                    style_3 = self.img_transform(torch.transpose(style_3,1,3).squeeze().unsqueeze(0),style_size)
                project_grayscale = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
                assert project_grayscale.weight.shape == (3, 3, 1, 1)
                # this is BGR to YUV, then setting U & V to zero, then back from YUV to BGR
                vals = torch.tensor(
                   [[0.0722,0.7152,0.212600],
                   [0.0722,0.7152,0.212600],
                   [0.0722,0.7152,0.212600],], dtype=project_grayscale.weight.dtype)
                with torch.no_grad():
                    project_grayscale.weight.copy_(vals.view(3, 3, 1, 1))
                project_grayscale = project_grayscale.to(device)
                # The "target" image to store outcome
                if init_image is None:
                    target = torch.rand_like(content_).to(dtype).requires_grad_(True).to(device)
                else:
                    target_ = self.img_transform(torch.transpose(init_image.to(device),1,3).squeeze().unsqueeze(0),output_size,(content_.size(2),content_.size(3)))
                    target = target_.clone().to(dtype)
                if "yes"==force_grayscale:
                    target = project_grayscale(target)
                target = target.contiguous().detach() #without this, L-BFGS dies
                target = target.requires_grad_(True) #and once again, because once isn't enough
                global CONTENT_WEIGHT
                global STYLE_WEIGHT
                global TV_WEIGHT
                global FINAL_LOSS
                CONTENT_WEIGHT = float(content_weight)
                STYLE_WEIGHT = float(style1_weight)+float(style2_weight)+float(style3_weight)
                TV_WEIGHT = float(total_variation_weight)
                # Precompute content features, style features, and style gram matrices.
                content_features = get_features(content_, model, CONTENT_LAYERS_DEFAULT)
                style1_features = get_features(style, model, STYLE_LAYERS_DEFAULT)
                if style_2 is None:
                    style2_features = None
                else:
                    style2_features = get_features(style_2, model, STYLE_LAYERS_DEFAULT)
                if style_3 is None:
                    style3_features = None
                else:
                    style3_features = get_features(style_3, model, STYLE_LAYERS_DEFAULT)
                style_grams = {
                    layer: gram_matrix(style1_features[layer],True) * ( float(style1_weight) / STYLE_WEIGHT )
                    for layer in style1_features
                }
                if style2_features is not None:
                    for layer in style1_features:
                        style_grams[layer] += gram_matrix(style2_features[layer],True) * ( float(style2_weight) / STYLE_WEIGHT )
                if style3_features is not None:
                    for layer in style3_features:
                        style_grams[layer] += gram_matrix(style3_features[layer],True) * ( float(style3_weight) / STYLE_WEIGHT )
                if optimizer == 'L-BFGS':
                    optimizer_ = optim.LBFGS([target], max_iter=iterations, line_search_fn=None, tolerance_grad = -1.0, tolerance_change = -1.0)
                    progressbar = tqdm(range(iterations))
                    pbar_iter = iter(progressbar)
                    comfy_progressbar = comfy.utils.ProgressBar(len(progressbar))
                    def closure():
                        if "yes"==force_grayscale:
                            target_ = project_grayscale(target)
                        else:
                            target_ = target
                        target_features = get_features(target_, model)
                        content_loss = content_loss_func(target_features, content_features, "enabled"==normalize_gradients).to(dtype)
                        style_loss = style_loss_func(target_features, style1_features, style_grams, "enabled"==normalize_gradients).to(dtype)
                        tvm = TVLossModule()
                        tv_loss = tvm(target).to(dtype)
                        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss
                        global FINAL_LOSS
                        FINAL_LOSS = total_loss
                        if torch.is_grad_enabled():
                            optimizer_.zero_grad(set_to_none=True)
                        if total_loss.requires_grad:
                            total_loss.backward()
                        progressbar.set_postfix_str(
                            f'total_loss={total_loss.item():.2f} '
                            f'content_loss={content_loss.item():.2f} '
                            f'style_loss={style_loss.item():.2f} '
                            f'tv_loss={tv_loss.item():.2f} '
                        )
                        next(pbar_iter)
                        comfy_progressbar.update(1)
                        return total_loss
                    optimizer_.step(closure)
                else:
                    if optimizer == 'ADAM':
                        optimizer_ = optim.Adam([target], lr=learning_rate_option)
                    progressbar = tqdm(range(iterations))
                    comfy_progressbar = comfy.utils.ProgressBar(len(progressbar))
                    for _ in progressbar:
                        optimizer_.zero_grad(set_to_none=True)
                        if "yes"==force_grayscale:
                            target_ = project_grayscale(target)
                        else:
                            target_ = target
                        target_features = get_features(target_, model)
                        content_loss = content_loss_func(target_features, content_features, "enabled"==normalize_gradients)
                        style_loss = style_loss_func(target_features, style1_features, style_grams, "enabled"==normalize_gradients)
                        tvm = TVLossModule()
                        tv_loss = tvm(target).to(dtype)
                        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss
                        global FINAL_LOSS
                        FINAL_LOSS = total_loss
                        total_loss.backward(retain_graph=True) # do we need `retain_graph=True`?
                        optimizer_.step()
                        progressbar.set_postfix_str(
                            f'total_loss={total_loss.item():.2f} '
                            f'content_loss={content_loss.item():.2f} '
                            f'style_loss={style_loss.item():.2f} '
                            f'tv_loss={tv_loss.item():.2f} '
                        )
                        comfy_progressbar.update(1)
                if "yes"==force_grayscale:
                    target = project_grayscale(target)
                target = self.inv_img_transform(target.detach().cpu())
                target = torch.transpose(target,1,3)
                return (target,FINAL_LOSS.tolist(),)

NODE_CLASS_MAPPINGS = {
    "NeuralStyle": NeuralStyle
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuralStyle": "NeuralStyle"
}
