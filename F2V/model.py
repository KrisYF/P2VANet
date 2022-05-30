import torch
import torch.nn as nn

"""
采用ResNet18进行特征提取
"""


class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        self.audio = nn.Sequential(
            nn.Conv2d(1, 3, 5, (2, 1), 1),
            # nn.Dropout2d(0.4),
            nn.AvgPool2d(kernel_size=(3, 5), stride=2, ceil_mode=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
            nn.Conv2d(3, 5, (3, 6), 2, 1),
            nn.BatchNorm2d(5),
            # nn.Dropout2d(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=False),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),

            nn.Conv2d(5, 8, 3, 1, 1),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            # nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.AvgPool2d(kernel_size=(4, 5), stride=1, ceil_mode=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
        )
        self.frame = nn.Sequential(
            #
            nn.Conv2d(3, 3, 7, 2, 3),
            # nn.Dropout2d(0.4),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
            # nn.Dropout2d(0.2),
            nn.Conv2d(3, 5, 5, 2, 2),
            # nn.Dropout2d(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),

            nn.Conv2d(5, 8, 2, 1, 4),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            # nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
        )

    def forward(self, f, a1, a2):
        f = self.frame(f)
        a1 = self.audio(a1)
        a2 = self.audio(a2)
        return f, a1, a2


"""
对提取的特征进行编码解码
"""


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder_input = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2, True)
        )
        self.encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(3200, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Tanh()
        )
        """
        计算均值和方差
        """
        self.fc_mu = nn.Linear(128, 128)
        self.fc_var = nn.Linear(128, 128)

        self.decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, 3200),
            nn.BatchNorm1d(3200),
            nn.LeakyReLU(0.2, True),
        )
        self.decoder_output = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            # nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2, True)
        )

    def encode(self, x):
        x = self.encoder_input(x)
        x = x.view(x.size(0), -1)
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return result, mu, log_var

    """
    重参数
    """

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, x):
        result = self.decoder(x)
        result = result.view(-1, 32, 10, 10)
        result = self.decoder_output(result)
        return result

    def forward(self, x):
        result, mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), x, result, z, mu, log_var


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(128, 2),
        )

    def forward(self, f, a1, a2):
        out_f = self.trans(f)
        out_a1 = self.trans(a1)
        out_a2 = self.trans(a2)
        return out_f, out_a1, out_a2


class Class(nn.Module):
    def __init__(self):
        super(Class, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 3, 2),
            nn.BatchNorm1d(2)
        )

    def forward(self, f, a1, a2):
        out = torch.cat([f, a1, a2], dim=1)
        return self.trans(out)


class Nation(nn.Module):
    def __init__(self):
        super(Nation, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 3),
            nn.BatchNorm1d(3)
        )

    def forward(self, f, a1, a2):
        out_f = self.trans(f)
        out_a1 = self.trans(a1)
        out_a2 = self.trans(a2)
        return out_f, out_a1, out_a2


class Gender(nn.Module):
    def __init__(self):
        super(Gender, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            nn.BatchNorm1d(2)
        )

    def forward(self, f, a1, a2):
        out_f = self.trans(f)
        out_a1 = self.trans(a1)
        out_a2 = self.trans(a2)
        return out_f, out_a1, out_a2


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self):
        super(MultiHeadCrossAttention, self).__init__()
        self.scale = 1 ** -0.5
        self.to_q = nn.Linear(128, 128, bias=False)
        self.to_kv = nn.Linear(512, 512 * 2, bias=False)

        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(128, 128)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, complement):
        # x [50, 128]
        B_x, N_x = x.shape  # 50, 128

        x_copy = x  # 50 * 128

        complement = torch.cat([x.float(), complement.float()], 1)  # 50 * 512

        B_c, N_c = complement.shape  # 50, 512

        # q [50, 1, 128, 1]
        q = self.to_q(x).reshape(B_x, N_x, 1, 1).permute(0, 2, 1, 3)
        # kv [2, 50, 1, 512, 1]
        kv = self.to_kv(complement).reshape(B_c, N_c, 2, 1, 1).permute(2, 0, 3, 1, 4)

        # 50 * 1 * 512 * 1
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 50 * 1 * 128 * 512
        attn = attn.softmax(dim=-1)  # 50 * 1 * 128 * 512
        attn = self.attn_drop(attn)  # 50 * 1 * 128 * 512

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x)  # 50 * 128

        x = x + x_copy

        x = self.proj(x)
        x = self.proj_drop(x)  # 50 * 128
        return x


class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        x = x.float()
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


"""
运用Transformer结构将音频特征和人脸属性融合
"""


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, mlp_ratio=1., proj_drop=0.3, act_layer=nn.GELU, norm_layer1=nn.LayerNorm,
                 norm_layer2=SwitchNorm1d):
        super(CrossTransformerEncoderLayer, self).__init__()
        self.x_norm1 = norm_layer1(128)
        self.c_norm1 = norm_layer2(384)

        self.attn = MultiHeadCrossAttention()

        self.x_norm2 = norm_layer1(128)

        mlp_hidden_dim = int(128 * mlp_ratio)
        self.mlp = Mlp(in_features=128, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x, complement):
        # x: 50 * 128
        # complement: 50 * 512
        x = self.x_norm1(x)  # 50 * 128
        complement = self.c_norm1(complement)  # 50 * 384

        x = x + self.drop1(self.attn(x, complement))  # 50 * 128
        x = x + self.drop2(self.mlp(self.x_norm2(x)))  # 50 * 128
        return x


class MultiHeadCrossAttentionVisual(nn.Module):
    def __init__(self):
        super(MultiHeadCrossAttentionVisual, self).__init__()
        self.scale = 1 ** -0.5
        self.to_q = nn.Linear(128, 128, bias=False)
        self.to_kv = nn.Linear(140, 140 * 2, bias=False)

        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(128, 128)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, complement):
        B_x, N_x = x.shape

        x_copy = x

        complement = torch.cat([x.float(), complement.float()], 1)

        B_c, N_c = complement.shape

        q = self.to_q(x).reshape(B_x, N_x, 1, 1).permute(0, 2, 1, 3)

        kv = self.to_kv(complement)

        kv = kv.reshape(B_c, N_c, 2, 1, 1).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x)

        x = x + x_copy

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


"""
运用Transformer结构将人脸特征和人脸属性融合
"""


class Visual(nn.Module):
    def __init__(self, mlp_ratio=1., proj_drop=0.3, act_layer=nn.GELU, norm_layer1=nn.LayerNorm,
                 norm_layer2=SwitchNorm1d):
        super(Visual, self).__init__()
        self.x_norm1 = norm_layer1(128)
        self.c_norm1 = norm_layer2(12)

        self.attn = MultiHeadCrossAttentionVisual()

        self.x_norm2 = norm_layer1(128)

        mlp_hidden_dim = int(128 * mlp_ratio)
        self.mlp = Mlp(in_features=128, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x, complement):
        x = self.x_norm1(x)
        complement = self.c_norm1(complement)

        x = x + self.drop1(self.attn(x, complement))
        x = x + self.drop2(self.mlp(self.x_norm2(x)))
        return x
