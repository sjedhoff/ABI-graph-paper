# GraphTransformer MeanPooling

import keras
import math
from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable
from bayesflow.networks import SummaryNetwork
from bayesflow.networks.transformers.pma import PoolingByMultiHeadAttention


class MultiHeadGraphAttention(keras.Layer):
    """Multi-head self-attention constrained by the adjacency (with self-loops)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        kernel_initializer: str = "glorot_uniform",
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias

        # Projections
        self.wq = keras.layers.Dense(embed_dim, use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.wk = keras.layers.Dense(embed_dim, use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.wv = keras.layers.Dense(embed_dim, use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.wo = keras.layers.Dense(embed_dim, use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x: Tensor, adj: Tensor, training: bool = False, node_mask: Tensor | None = None) -> Tensor:
        """
        x: (B, N, F)  node features
        adj: (B, N, N) adjacency (0/1), we add self-loops internally
        node_mask: (B, N) optional boolean/0-1 for valid nodes
        returns: (B, N, F)
        """
        B, N, _ = keras.ops.shape(x)

        # Add self-loops
        eye = keras.ops.eye(N, dtype=adj.dtype)
        adj_sl = adj + eye  # (B, N, N) + (N, N) -> broadcast over batch

        # Project to Q,K,V and reshape to heads
        q = self.wq(x)  # (B, N, E)
        k = self.wk(x)
        v = self.wv(x)

        def _split_heads(t):
            # (B, N, E) -> (B, H, N, D)
            t = keras.ops.reshape(t, (B, N, self.num_heads, self.head_dim))
            return keras.ops.transpose(t, (0, 2, 1, 3))

        qh, kh, vh = _split_heads(q), _split_heads(k), _split_heads(v)

        # Scaled dot-product attention
        # scores: (B, H, N, N)
        scores = keras.ops.matmul(qh, keras.ops.transpose(kh, (0, 1, 3, 2)))
        scale = math.sqrt(float(self.head_dim))
        scores = scores / scale

        # Build mask from adjacency (+ optional node mask)
        # base graph mask (B, 1, N, N)
        gmask = keras.ops.expand_dims(adj_sl > 0, axis=1)

        if node_mask is not None:
            # ensure boolean
            nmask = node_mask > 0
            # (B, 1, N, 1) & (B, 1, 1, N) -> (B, 1, N, N)
            row_ok = keras.ops.expand_dims(keras.ops.expand_dims(nmask, axis=1), axis=3)
            col_ok = keras.ops.expand_dims(keras.ops.expand_dims(nmask, axis=1), axis=2)
            gmask = keras.ops.logical_and(gmask, keras.ops.logical_and(row_ok, col_ok))

        # Apply mask by adding large negative to disallowed positions
        very_neg = keras.ops.full_like(scores, -1e9)
        scores = keras.ops.where(gmask, scores, very_neg)

        attn = keras.ops.softmax(scores, axis=-1)  # (B, H, N, N)
        attn = self.dropout(attn, training=training)

        out = keras.ops.matmul(attn, vh)  # (B, H, N, D)
        # Merge heads: (B, N, E)
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (B, N, self.embed_dim))
        return self.wo(out)


class GraphSetAttentionBlock(keras.Layer):
    """SetAttentionBlock variant whose attention is restricted by the graph adjacency."""
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_depth: int = 2,
        mlp_width: int = 128,
        dropout: float = 0.05,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.dropout_rate = dropout
        self.mlp_activation = mlp_activation
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.layer_norm_flag = layer_norm

        self.input_proj = keras.layers.Dense(embed_dim, use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.attn = MultiHeadGraphAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
        )
        self.drop1 = keras.layers.Dropout(dropout)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6) if layer_norm else keras.layers.Lambda(lambda t: t)

        # FFN
        mlp_layers = []
        for d in range(mlp_depth - 1):
            mlp_layers += [
                keras.layers.Dense(mlp_width, activation=mlp_activation, kernel_initializer=kernel_initializer, use_bias=use_bias),
                keras.layers.Dropout(dropout),
            ]
        mlp_layers += [keras.layers.Dense(embed_dim, activation=None, kernel_initializer=kernel_initializer, use_bias=use_bias)]
        self.mlp = keras.Sequential(mlp_layers)
        self.drop2 = keras.layers.Dropout(dropout)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6) if layer_norm else keras.layers.Lambda(lambda t: t)

    def call(self, x: Tensor, adj: Tensor, training: bool = False, node_mask: Tensor | None = None) -> Tensor:
        # Pre-proj to model dim
        h = self.input_proj(x)
        # Attention (pre-norm)
        h_norm = self.norm1(h)
        h_attn = self.attn(h_norm, adj=adj, training=training, node_mask=node_mask)
        h = h + self.drop1(h_attn, training=training)

        # FFN
        h_norm = self.norm2(h)
        h_ffn = self.mlp(h_norm, training=training)
        h = h + self.drop2(h_ffn, training=training)
        return h


@serializable("bayesflow.networks", disable_module_check=True)
class GraphTransformerMeanPooling(SummaryNetwork):
    """
    SetTransformer-like encoder for graphs.
    - Input matches your simulator: (B, N, 1 + N) where the first feature is the node label x,
      and the remaining N features are the adjacency row.
    - Attention blocks are graph-aware (restricted by adjacency).
    - Pooling is the same PMA used by SetTransformer.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        embed_dims: tuple = (64, 64),
        num_heads: tuple = (4, 4),
        mlp_depths: tuple = (2, 2),
        mlp_widths: tuple = (128, 128),
        num_seeds: int = 1,
        dropout: float = 0.05,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        # PMA options
        seed_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not (len(embed_dims) == len(num_heads) == len(mlp_depths) == len(mlp_widths)):
            raise ValueError("embed_dims, num_heads, mlp_depths, mlp_widths must have the same length.")

        self.blocks = []
        for i in range(len(embed_dims)):
            self.blocks.append(
                GraphSetAttentionBlock(
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_depth=mlp_depths[i],
                    mlp_width=mlp_widths[i],
                    dropout=dropout,
                    mlp_activation=mlp_activation,
                    kernel_initializer=kernel_initializer,
                    use_bias=use_bias,
                    layer_norm=layer_norm,
                )
            )

        self.agg = keras.layers.GlobalAveragePooling1D()

        self.flatten = keras.layers.Flatten() if num_seeds > 1 else None
        self.out = keras.layers.Dense(summary_dim, activation=None)
        self.summary_dim = summary_dim
        self.num_seeds = num_seeds

        # Keep for config
        self.embed_dims = tuple(embed_dims)
        self.num_heads = tuple(num_heads)
        self.mlp_depths = tuple(mlp_depths)
        self.mlp_widths = tuple(mlp_widths)
        self.dropout = dropout
        self.mlp_activation = mlp_activation
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.layer_norm = layer_norm
        self.seed_dim = seed_dim

    def call(self, inputs: Tensor, training: bool = False, node_mask: Tensor | None = None, **kwargs) -> Tensor:
        """
        inputs: (B, N, 1 + N) from your simulator (first column = node feature, rest = adjacency row).
        node_mask: optional (B, N) for variable-sized graphs/padding; combined with adjacency.
        """
        B, N, T = keras.ops.shape(inputs)
        F = T - N  # number of node features in the left block; for your simulator, F == 1

        x = inputs[..., :F]         # (B, N, F)
        a = inputs[..., F:]         # (B, N, N)

        h = x
        for blk in self.blocks:
            h = blk(h, adj=a, training=training, node_mask=node_mask)

        # MeanPooling
        g = self.agg(h) 

        return self.out(g)

    def get_config(self):
        return dict(
            summary_dim=self.summary_dim,
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            mlp_depths=self.mlp_depths,
            mlp_widths=self.mlp_widths,
            num_seeds=self.num_seeds,
            dropout=self.dropout,
            mlp_activation=self.mlp_activation,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
            layer_norm=self.layer_norm,
            seed_dim=self.seed_dim,
        )
