import keras
from bayesflow.networks import SummaryNetwork
from bayesflow.networks.transformers.pma import PoolingByMultiHeadAttention
from bayesflow.utils.serialization import serializable


class GCNConv(keras.Layer):
    def __init__(self, units, activation="elu", initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.initializer = initializer

    def build(self, input_shape):
        x_shape, _ = input_shape
        self.kernel = self.add_weight(
            shape=(x_shape[-1], self.units),
            initializer=self.initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        x, a = inputs  # x: (B, N, F), a: (B, N, N)

        # Add self-loops
        a_tilde = a + keras.ops.eye(a.shape[-1])

        # Degree matrix and safe inverse sqrt
        degree = keras.ops.sum(a_tilde, axis=-1, keepdims=True)
        epsilon = 1e-8
        d_inv_sqrt = 1.0 / keras.ops.sqrt(degree + epsilon)

        a_hat = d_inv_sqrt * a_tilde * keras.ops.transpose(d_inv_sqrt, (0, 2, 1))

        xw = keras.ops.matmul(x, self.kernel)
        output = keras.ops.matmul(a_hat, xw) + self.bias

        return self.activation(output) if self.activation else output


@serializable("custom")
class GCNMHA(SummaryNetwork):
    """
    Initializes the GNNEncoder with an arbitrary number of GCN layers.

    Parameters
    ----------
    gcn_units : tuple of int, optional
        Tuple specifying the number of units in each GCNConv layer. Defaults to (32, 32).t
    summary_dim : int, optional
        The dimensionality of the final dense output layer. Defaults to 64.
    activation : str or None, optional
        Activation function to use in each GCNConv layer. Defaults to "elu".
    dropout : float or None, optional      
        Dropout rate (0 ≤ dropout < 1) applied *after* each GCNConv layer. Defaults to 0.05.
    """
    def __init__(self, gcn_units: tuple = (32, 32), summary_dim: int = 64, activation: str | None = "elu", initializer: str | None = "glorot_uniform",
     dropout: float = 0.05, num_heads=4, mlp_depth=2, mlp_width=128, num_seeds=1):
        super().__init__()
        self.gcn_layers = [GCNConv(units, activation, initializer) for units in gcn_units]
        self.dropouts = [keras.layers.Dropout(dropout) for _ in gcn_units]

        # Pooling
        embed_dim = gcn_units[-1]
        self.pooling_by_attention = PoolingByMultiHeadAttention(
            num_heads=num_heads, embed_dim=embed_dim,
            mlp_depth=mlp_depth, mlp_width=mlp_width,
            num_seeds=num_seeds, seed_dim=embed_dim,
            dropout=dropout, mlp_activation="gelu", layer_norm=True
        )

        self.dense = keras.layers.Dense(summary_dim)
        self.summary_dim = summary_dim
        self.gcn_units = gcn_units
        self.activation = activation
        self.dropout = dropout

    def call(self, inputs, training=False, **kwargs):
        """
        Applies the GNN encoder to batched input graphs.

        Expects as input a batched tensor where the last axis concatenates
        node features and the adjacency matrix for each graph.

        Parameters
        ----------
        inputs : Tensor of shape (batch, n_nodes, n_features + n_nodes)
            Batched input tensor containing the node features (first n_features columns)
            and the adjacency matrix (remaining n_nodes columns) for each graph.

        Returns
        -------
        output : Tensor of shape (batch, summary_dim)
            Encoded graph-level representations.
        """
        batch_size, num_nodes, total_features = inputs.shape
        num_features = total_features - num_nodes

        x = inputs[..., :num_features]     # (batch, n_nodes, 1)
        a = inputs[..., num_features:]     # (batch, n_nodes, n_nodes)

        # Apply GCN layers
        for gcn_layer, dropout_layer in zip(self.gcn_layers, self.dropouts):
            x = gcn_layer([x, a])
            x = dropout_layer(x, training=training)

        x = self.pooling_by_attention(x, training=training, **kwargs)
        return self.dense(x)


    def get_config(self):
        return {
            "gcn_units": self.gcn_units,
            "summary_dim": self.summary_dim,
            "activation": self.activation,
            "dropout": self.dropout,
        }
