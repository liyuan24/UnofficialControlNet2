import torch as th
from torch import nn

from ldm.modules.attention import zero_module, SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResNetBlock, Downsample
from ldm.modules.diffusionmodules.util import timestep_embedding, linear, conv_nd


class ControlledUNetModel(UNetModel):
    """
    This is the UNet model which also takes control outputs as inputs in the middle block and decoder
    """

    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        hs = []
        with th.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
        if control is not None:
            h += control.pop()
        for i, module in enumerate(self.output_blocks):
            # control is added to the model by adding with encoder output of the controlled unet
            h = th.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.type)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolution,
            dropout=0.,
            channel_mult=(1, 2, 4, 8),
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=None,
            use_linear_in_transformer=False,
    ):
        super().__init__()
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
        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
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
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, hint_channels, 16, 16, padding=1),
            nn.SiLU(),
            conv_nd(dims, hint_channels, 16, 32, padding=1),
            nn.SiLU(),
            conv_nd(dims, hint_channels, 32, 32, padding=1),
            nn.SiLU(),
            conv_nd(dims, hint_channels, 32, 96, padding=1),
            nn.SiLU(),
            conv_nd(dims, hint_channels, 96, 96, padding=1),
            nn.SiLU(),
            conv_nd(dims, hint_channels, 96, 256, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
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
                self.zero_convs.append(self.make_zero_conv(ch))
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, dims=dims)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
                self.zero_convs.append(self.make_zero_conv(ch))
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
        self.middle_block_output = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        # for the control process, it will take timestep embedding and context(prompt) as input
        # the output is [b, model_ch, image_size, image_size]
        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h))
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_output(h))
        return outs


