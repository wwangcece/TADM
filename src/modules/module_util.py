import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import sys
p = "src/modules"
sys.path.append(p)
import thops

# [-1, 1] STE
class Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, -1, 1)
        output = (input * 255.0).round() / 255.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input, train=True):
        return Quant.apply(input)

class Quanti(nn.Module):
    def __init__(self):
        super(Quanti, self).__init__()

    def forward(self, input, train=True):
        output = (input + 1.0) / 2.0
        output = torch.clamp(output, 0, 1)
        output = torch.round(output * 255.0) / 255.0
        output = 2 * output - 1
        return input - (input - output).detach()


# [-1, 1] Noise
class Quan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, train=True):
        x = (x + 1.0) / 2.0
        x = x * 255.0
        if train:
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            output = x + noise
            output = torch.clamp(output, 0, 255.0)
        else:
            output = x.round() * 1.0
            output = torch.clamp(output, 0, 255.0)
        return (output / 255.0 - 0.5) / 0.5


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode="bilinear", padding_mode="zeros"):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        if (self.bias != 0).any():
            self.inited = True
            return
        assert input.device == self.bias.device, (input.device, self.bias.device)
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, rev=False, offset=None):
        bias = self.bias

        if offset is not None:
            bias = bias + offset

        if not rev:
            return input + bias
        else:
            return input - bias

    def _scale(self, input, logdet=None, rev=False, offset=None):
        logs = self.logs

        if offset is not None:
            logs = logs + offset

        if not rev:
            input = input * torch.exp(
                logs
            )  # should have shape batchsize, n_channels, 1, 1
            # input = input * torch.exp(logs+logs_offset)
        else:
            input = input * torch.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply pixels
            """
            dlogdet = thops.sum(logs) * thops.pixels(input)
            if rev:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(
        self,
        input,
        rev=False,
        logdet=None,
        offset_mask=None,
        logs_offset=None,
        bias_offset=None,
    ):
        if not self.inited:
            self.initialize_parameters(input)

        if offset_mask is not None:
            logs_offset *= offset_mask
            bias_offset *= offset_mask
        # no need to permute dims as old version
        if not rev:
            # center and scale
            input = self._center(input, rev, bias_offset)
            input, logdet = self._scale(input, logdet, rev, logs_offset)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, rev, logs_offset)
            input = self._center(input, rev, bias_offset)
        return input


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()
            )
        )


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            # W = PL(U+diag(s))
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer(
                "p", torch.Tensor(np_p.astype(np.float32))
            )  # remains fixed
            self.register_buffer(
                "sign_s", torch.Tensor(np_sign_s.astype(np.float32))
            )  # the sign is fixed
            self.l = nn.Parameter(
                torch.Tensor(np_l.astype(np.float32))
            )  # optimized except diagonal 1
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))  # optimized
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, rev):
        # The difference in computational cost will become significant for large c, although for the networks in
        # our experiments we did not measure a large difference in wallclock computation time.
        if not self.LU:
            if not rev:
                # pixels = thops.pixels(input)
                # GPU version
                # dlogdet = torch.slogdet(self.weight)[1] * pixels
                # CPU version is 2x faster, https://github.com/didriknielsen/survae_flows/issues/5.
                dlogdet = (
                    torch.slogdet(self.weight.to("cpu"))[1] * thops.pixels(input)
                ).to(self.weight.device)
                weight = self.weight.view(self.w_shape[0], self.w_shape[1], 1, 1)
            else:
                dlogdet = 0
                weight = (
                    torch.inverse(self.weight.double())
                    .float()
                    .view(self.w_shape[0], self.w_shape[1], 1, 1)
                )

            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(
                self.sign_s * torch.exp(self.log_s)
            )
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not rev:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, rev=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, rev)
        if not rev:
            z = F.conv2d(input, weight)  # fc layer, ie, permute channel
            if logdet is not None:
                logdet = logdet + dlogdet
            return z
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]

            out = (
                F.conv2d(
                    x, self.haar_weights, bias=None, stride=2, groups=self.channel_in
                )
                / 4.0
            )
            out = out.reshape(
                [x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2]
            )
            out = torch.transpose(out, 1, 2)
            out = out.reshape(
                [x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2]
            )
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(
                out, self.haar_weights, bias=None, stride=2, groups=self.channel_in
            )


class Convkxk(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, k_size, padding=(k_size - 1) // 2
        )
        self.conv2 = nn.Conv2d(
            out_channels, in_channels, k_size, padding=(k_size - 1) // 2
        )

    def forward(self, input, rev=False):
        if not rev:
            z = self.conv1(input)
            # print(z.size())
            return z
        else:
            z = self.conv2(input)
            return z
