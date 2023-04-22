from abc import abstractmethod

from torch import nn
import torch as th
import torch.nn.functional as F

from ldm.modules.attention import SpatialTransformer, zero_module
from ldm.util import default
from util import linear, conv_nd, group_norm, SiLU, timestep_embedding


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding
    :param in_channels: channels in the input tensor
    :param model_channels: base channel count for the model
    :param out_channels: channels in the output tensor
    :param num_res_blocks: number of ResNet blocks in each level
    :param attention_resolution: a collection of downsample rates at which attention will take place. Maybe a set, list,
        or tuple. For example, if it contains 2, then at 2x downsampling, attention will be used
    :param dropout: the dropout ratio
    :param channel_mult: a list of multipliers on model_channels for each level
    :param dims: determine if the signal is 1D, 2D or 3D
    :param num_heads: number of heads in each attention layer
    :param num_head_channels: number of channels per head in attention layer
    :param use_spatial_transformer: whether to use spatial transformer
    :param transformer_depth: number of basic transformer blocks in the spatial transformer layer
    :param context_dim: the dimension of the context encoding
    :param use_linear_in_transformer: whether to use linear layer to adjust the number of channels. If False,
        it will use convolution
    :param use_checkpoint: use gradient checkpointing to reduce memory usage
    :param use_fp16: set the data type as torch.float16
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolution,
            dropout=0,
            channel_mult=(1, 2, 4, 4),
            dims=2,
            num_heads=-1,
            num_head_channels=-1,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=None,
            use_linear_in_transformer=False,
            use_checkpoint=False,
            use_fp16=False
    ):
        super.__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include te dimension of the cross-attention ' \
                                            'conditioning...'
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use spatial transformer for your corss-attention ' \
                                            'conditioning'
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_head or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_head or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks as an int or as a list of tuple with the same size as "
                                 "channel_mult")
            self.num_res_blocks = num_res_blocks
        self.attention_resolution = attention_resolution
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3,
                                                paddings=1))
            ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResNetBlock(
                        ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims
                    )
                ]
                ch = mult * model_channels
                # add SpatialTransformer block
                if ds in attention_resolution:
                    if num_heads == -1:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    else:
                        dim_head = ch // num_heads

                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, context_dim=context_dim,
                                           use_linear=use_linear_in_transformer,
                                           use_checkpoint=use_checkpoint
                                           )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, dims=dims)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
        if num_heads == -1:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        else:
            dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResNetBlock(ch, time_embed_dim, dropout, dims=dims),
            SpatialTransformer(ch, num_heads, dim_head, context_dim, use_linear=use_linear_in_transformer,
                               use_checkpoint=use_checkpoint),
            ResNetBlock(ch, time_embed_dim, dropout, dims=dims)
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                # corresponding input block channels
                ich = input_block_chans.pop()
                layers = [
                    ResNetBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolution:
                    if num_heads == -1:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    else:
                        dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, context_dim, use_linear=use_linear_in_transformer,
                                           use_checkpoint=use_checkpoint))
                if level and i == self.num_res_blocks[level]:
                    layers.append(Upsample(ch, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            group_norm(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps=None, context=None, **kwargs):
        """
        Apply encoder and decoder to the input batch
        :param x: an [N, C, H, W] tensor
        :param timesteps: a 1-D tensor of timesteps with size N
        :param context: conditional input used in cross attention in Spatial Transformer
        :return: an [N, C, H, W] tensor
        """
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.type)
        return self.out(h)



class Downsample(nn.Module):
    def __init__(self, channels, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels, dims=2, out_channels=None):
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        self.dims = dims
        self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embedding as a second parameter
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        :param x: a tensor
        :param emb: timestep embedding
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as a second parameter
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class ResNetBlock(TimestepBlock):
    """
    A residual network block that can optinally change the number of channels
    :param channels: the number of input_channels
    :param embed_channels: the number of channels for timestep embedding
    :param dropout: the rate of dropout
    :param out_channels: if specified, the number of output channels
    :param dims: the convolution dimension
    """

    def __init__(self, channels, embed_channels, dropout, out_channels=None, dims=2):
        super().__init__()
        self.channels = channels
        self.embed_channels = embed_channels
        self.dropout = dropout
        self.out_channels = default(out_channels, channels)
        # the transformation layer for the image input
        self.in_layers = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # the transformation layer for timestep embedding
        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_channels, self.out_channels)
        )

        # the transformation layer for the concatenation of timestep embedding and image
        self.out_layer = nn.Sequential(
            group_norm(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))
        )

        # skip connection layer
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        x_in = x
        x = self.in_layers(x)
        emb_out = self.embed_layer(emb).type(x.dtype)
        # the emb_out shape is [b, out_channels, 1, 1]
        # the shape of x is [b, out_channels, h, w]
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x = x + emb_out
        x = self.out_layer(x)
        return x + self.skip_connection(x_in)
