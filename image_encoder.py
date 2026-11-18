"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import math
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        #
        # The feature axis (last axis) is cut into H consecutive slices of
        # head_dim values, one head per slice. Reshape splits D into (H, hd)
        # without copying or reordering, so head h keeps features
        # h*hd ... h*hd+hd-1 of every token, exactly as the checkpoint stores
        # them. The new axis order is [B,N,H,hd]; the head axis then moves in
        # front of the token axis to obtain the [B,H,N,hd] layout used by the
        # batched matrix products. Token order and feature order inside a head
        # are both preserved.
        batch, tokens, width = x.shape
        split = x.reshape(batch, tokens, self.heads, self.head_dim)
        return split.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        #
        # Every query row is compared with every key row through a dot
        # product, so the key array must present (hd, N) per (image, head).
        # k is 4D here, therefore only its final two axes are exchanged; a
        # plain k.T would also swap the batch and head axes and mix unrelated
        # images. Dividing by sqrt(head_dim) keeps the score scale independent
        # of the head width before the softmax-like coefficient conversion.
        head_scale = np.float32(math.sqrt(self.head_dim))
        scores = (q @ k.swapaxes(-1, -2)) / head_scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        #
        # coefficients is [B,H,N,N] and v is [B,H,N,hd], so one matrix product
        # gives every token a weighted average of the value rows of its own
        # head, producing [B,H,N,hd]. Merging reverses split_heads: the head
        # axis returns next to the head_dim axis and both are fused back into
        # the single width axis, which restores the token order and the
        # original feature order inside each token.
        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        #
        # out_weight is stored [out,in] like every attention and MLP linear
        # layer of this checkpoint, so the transposed form maps the merged
        # width to the output width. The bias is one vector and broadcasts
        # over the batch and token axes.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        #
        # The attention branch only produces a correction, so it is added back
        # to the un-normalized block input. Element-wise addition of two
        # [B,N,D] arrays keeps the shape and the token order unchanged.
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        #
        # The MLP is applied independently to every token row: the first
        # linear layer expands 768 to 3072 per token, the exact erf-based GELU
        # is applied element-wise, and the second linear layer contracts back
        # to 768. Both stored weights are [out,in], hence the transposes.
        # This block has no LayerScale parameter, so the second residual is a
        # plain addition and the returned array keeps shape [B,N,D].
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        #
        # 224 = 7 * 32, so each spatial axis carries both the patch index and
        # the position inside a patch. Reshaping to [B,C,7,32,7,32] separates
        # them without moving any value; the axis order is then
        # (B, C, row_of_patch, y_in_patch, col_of_patch, x_in_patch).
        # Transposing to (B, patch_row, patch_col, C, y, x) puts the two patch
        # axes first, which fixes the required left-to-right then top-to-bottom
        # patch order, and keeps C before H and W inside each patch. The final
        # reshape fuses (C,y,x) into the 3*32*32=3072 patch vector.
        # A single reshape(B,49,3072) would instead read 32 consecutive pixel
        # rows, which is not the requested patch order.
        batch = images.shape[0]
        grid = images.reshape(batch, 3, 7, 32, 7, 32)
        grid = grid.transpose(0, 2, 4, 1, 3, 5)
        return grid.reshape(batch, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        #
        # conv1.weight was flattened to [768,3072], i.e. [out,in], so the
        # transpose maps each 3072-value patch vector to its 768-value
        # embedding. This is a linear map shared by all 49 patches of all
        # images, and the checkpoint supplies no bias term for it.
        projected = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        #
        # The CLS row becomes token 0 and the 49 patch rows follow it, giving
        # 50 rows per image. positional_embedding is [50,768] and holds one
        # learned vector per token position, so broadcasting adds position j to
        # row j of every image in the batch.
        tokens = np.concatenate((cls, projected), axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        #
        # Row 0 of the token axis is the CLS row, the only row that has seen
        # the whole image through attention, so it is selected as the image
        # summary and the other 49 patch rows are dropped. ln_post is a
        # separate learned Norm applied to that single row per image, so the
        # input of this Norm is [B,768] and it normalizes along the feature
        # axis. visual.proj is stored [in,out] = [768,512] and is therefore
        # used without a transpose. unit_vector then rescales each image row
        # to length one, which gives the cosine-ready vectors the final
        # project needs.
        cls_token = tokens[:, 0, :]
        pooled = self.post_norm(cls_token)
        output = pooled @ self.projection

        return unit_vector(output)
