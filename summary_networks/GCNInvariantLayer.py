import keras
from bayesflow.networks import SummaryNetwork
from bayesflow.networks.deep_set.invariant_layer import InvariantLayer
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
class GCNInvariantLayer(SummaryNetwork):
    """
    GCN encoder with an invariant pooling layer.
    """

    def __init__(
        self,
        gcn_units: tuple = (32, 32),
        summary_dim: int = 64,
        activation: str | None = "elu",
        initializer: str | None = "glorot_uniform",
        dropout: float = 0.05,
        mlp_widths_pooling=(64, 64),
        **kwargs,
    ):
        super().__init__(**kwargs)

        # store all init args for serialization
        self.gcn_units = gcn_units
        self.summary_dim = summary_dim
        self.activation = activation
        self.initializer = initializer
        self.dropout = dropout
        self.mlp_widths_pooling = mlp_widths_pooling

        self.gcn_layers = [
            GCNConv(units, activation, initializer) for units in gcn_units
        ]
        self.dropouts = [keras.layers.Dropout(dropout) for _ in gcn_units]

        self.agg = InvariantLayer(
            mlp_widths_inner=mlp_widths_pooling,
            mlp_widths_outer=mlp_widths_pooling,
            activation="silu",
            kernel_initializer="he_normal",
            dropout=dropout,
            pooling="mean",
            spectral_normalization=False,
        )

        self.dense = keras.layers.Dense(summary_dim)

    def call(self, inputs, training=False, **kwargs):
        batch_size, num_nodes, total_features = inputs.shape
        num_features = total_features - num_nodes

        x = inputs[..., :num_features]     # (batch, n_nodes, F)
        a = inputs[..., num_features:]     # (batch, n_nodes, n_nodes)

        # Apply GCN layers
        for gcn_layer, dropout_layer in zip(self.gcn_layers, self.dropouts):
            x = gcn_layer([x, a])
            x = dropout_layer(x, training=training)

        x = self.agg(x, training=training)
        return self.dense(x)

    def get_config(self):
        # Start from parent config to keep Keras metadata
        base_config = super().get_config()
        base_config.update(
            {
                "gcn_units": self.gcn_units,
                "summary_dim": self.summary_dim,
                "activation": self.activation,
                "initializer": self.initializer,
                "dropout": self.dropout,
                "mlp_widths_pooling": self.mlp_widths_pooling,
            }
        )
        return base_config
