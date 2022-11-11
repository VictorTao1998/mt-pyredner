import torch
from torch.nn import functional as F

TRIVIAL_TERM = 1e-8

def pad_img(img, block_size):
    half = block_size // 2
    padding = torch.zeros((img.shape[0], half)).to(img)
    img = torch.cat([padding, img, padding], dim=1)
    padding = torch.zeros((half, img.shape[1])).to(img)
    img = torch.cat([padding, img, padding], dim=0)

    return img

def patch_mean(images, patch_shape):    
    """
    Computes the local mean of an image or set of images.

    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)

    Returns:
        Tensor same size as the image, with local means computed independently for each channel.
    """
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).bool()
    weights[~channel_selector] = 0

    result = conv(images, weights, padding='valid', bias=None)

    return result


def patch_std(image, patch_shape):
    """
    Computes the local standard deviations of an image or set of images.

    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)

    Returns:
        Tensor same size as the image, with local standard deviations computed independently for each channel.
    """
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2 + TRIVIAL_TERM).sqrt()


def channel_normalize(template):
    """
    Z-normalize image channels independently.
    """
    reshaped_template = template.clone().view(template.shape[0], template.shape[1], -1)
    reshaped_template = reshaped_template - reshaped_template.mean(dim=-1, keepdim=True)
    reshaped_template = torch.div(reshaped_template, (reshaped_template.std(dim=-1, keepdim=True, unbiased=False) + TRIVIAL_TERM))

    return reshaped_template.view_as(template)


class NCC(torch.nn.Module):
    """
    Computes the Zero-Normalized Cross-Correlation between an image and a template.
    """
    def __init__(self, template):
        super().__init__()

        _, channels, *template_shape = template.shape
        dimensions = len(template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)[:, None, ...]
        self.normalized_template = self.normalized_template.repeat(1, channels, 1, 1, 1)

        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).bool()
        self.normalized_template[:, ~channel_selector] = 0

        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template = torch.div(self.normalized_template, patch_elements)

        # Use grouped kernels
        self.normalized_template = torch.cat(self.normalized_template.unbind())

    def forward(self, inputs):
        batch_size, channel, h, w = inputs.shape
        inputs_reshaped = inputs.view(1, -1, h, w)
    
        result = self.conv_f(inputs_reshaped, self.normalized_template, padding='valid', groups=batch_size)
        result = result.view(batch_size, channel, result.shape[2], result.shape[3])
        std = patch_std(inputs, self.normalized_template.shape[1:])
        result = torch.div(result, std + TRIVIAL_TERM)

        return result

class SoftArgmax1D(torch.nn.Module):
    def __init__(self, base_index=0, step_size=1):
        super().__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        smax = self.softmax(x)
        end_index = self.base_index + x.shape[1] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size).to(x)

        return torch.matmul(smax, indices)

def block_matching(img_l, img_r, min_disp, max_disp, block_size, temperature, batch_size):
    # Compute matching cost array
    left_padded = pad_img(img_l, block_size)[..., None]
    left_padded = left_padded.permute(2, 0, 1)
    right_padded = pad_img(img_r, block_size)[..., None]
    right_padded = right_padded.permute(2, 0, 1)

    h, w = img_l.shape
    half = block_size // 2
    matching_cost = []
    for i in range(h // batch_size):
        kernels = []
        inputs = []
        for y in range(i * batch_size, (i + 1) * batch_size):
            for x in range(max_disp, w):
                kernel = left_padded[:, y:y+2*half+1, x:x+2*half+1]
                input_arr = right_padded[:, y:y+2*half+1, x-max_disp:x-min_disp+2*half+1]
                kernels.append(kernel)
                inputs.append(input_arr)
        kernels = torch.stack(kernels)
        inputs = torch.stack(inputs)
        ncc = NCC(kernels)
        result = ncc(inputs).squeeze().flip(1)
        assert result.isnan().any() == False # If true, increase TRIVIAL_TERM defined at top
        matching_cost.append(result)
    
    ## Height not divisible by batch size
    if h % batch_size != 0:
        kernels = []
        inputs = []
        for y in range((h // batch_size) * batch_size, h):
            for x in range(max_disp, w):
                kernel = left_padded[:, y:y+2*half+1, x:x+2*half+1]
                input_arr = right_padded[:, y:y+2*half+1, x-max_disp:x-min_disp+2*half+1]
                kernels.append(kernel)
                inputs.append(input_arr)
        kernels = torch.stack(kernels)
        inputs = torch.stack(inputs)
        ncc = NCC(kernels)
        result = ncc(inputs).squeeze().flip(1)
        assert result.isnan().any() == False # If true, increase TRIVIAL_TERM defined at top
        matching_cost.append(result)

    matching_cost = torch.cat(matching_cost, axis=0)

    # Softargmax
    softargmax = SoftArgmax1D(base_index=min_disp)
    disp = softargmax(matching_cost * temperature)

    # Reshape to img
    disp = disp.reshape(h, w - max_disp)

    return disp
