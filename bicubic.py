import torch
from torch import nn
from torch.nn import functional as F


class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return 0.0

    def bilinear_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bilinear
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            1 - abs_x
        else:
            return 0

    def __init__(self, factor=4, cuda=True, padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
                          for i in range(size)], dtype=torch.float32)
        k = k / torch.sum(k)
        # k = torch.einsum('i,j->ij', (k, k))
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0)
        self.cuda = '.cuda' if cuda else ''
        self.padding = padding
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        # x = torch.from_numpy(x).type('torch.FloatTensor')
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1.type('torch{}.FloatTensor'.format(self.cuda))
        filters2 = self.k2.type('torch{}.FloatTensor'.format(self.cuda))

        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(
                x, 2, 3), 1, 2)   # NHWC to NCHW

        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x, weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)

        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)

        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type('torch.ByteTensor'.format(self.cuda))
        else:
            return x


class BicubicDownsampleTargetSize(nn.Module):
    def __init__(self, target_size, cuda=True, padding='reflect'):
        super().__init__()
        self.target_size = target_size
        self.cuda = cuda
        self.padding = padding
        self.bicubic_downsamplers = {}

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        input_size = x.shape[1]
        #with_batch_idx = x.unsqueeze(0)
        if input_size > self.target_size:
            raise Exception("Input size must be between target_size/2 and target_size")
        mode = 'bilinear' if input_size >= self.target_size / 2 else 'area'
        test = BicubicDownsampleTargetSize.downsampling(x, (self.target_size, self.target_size), mode=mode)
        #test = test.squeeze(0)
        return test
        #target_factor = self.target_size / input_size
        #if target_factor not in self.bicubic_downsamplers:
        #    self.bicubic_downsamplers = BicubicDownSample(target_factor, self.cuda, self.padding)
        #downsampled = self.bicubic_downsamplers[target_factor].forward(x, nhwc, clip_round, byte_output)
        #return downsampled

    # https://discuss.pytorch.org/t/autogradable-image-resize/580/7
    @staticmethod
    def downsampling(x, size=None, scale_factor=None, mode='nearest'):
        import torch.nn.functional
        align_corners = True if mode == 'bilinear' else None
        downsampled = torch.nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
        return downsampled
        # define size if user has specified scale_factor
        if size is None: size = (int(scale_factor*x.size(2)), int(scale_factor*x.size(3)))
        # create coordinates
        h = torch.arange(0,size[0]) / (size[0]-1) * 2 - 1
        w = torch.arange(0,size[1]) / (size[1]-1) * 2 - 1
        # create grid
        grid = torch.zeros(size[0],size[1],2)
        grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
        grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
        # expand to match batch size
        grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
        if x.is_cuda: grid = grid.cuda()
        # do sampling
        return F.grid_sample(x, grid, mode=mode)

    @staticmethod
    def downsample_single(x, size=None, scale_factor=None, mode='nearest'):
        x = x.unsqueeze(0)
        x = BicubicDownsampleTargetSize.downsampling(x, size, scale_factor, mode)
        x = x.squeeze(0)
        return x
