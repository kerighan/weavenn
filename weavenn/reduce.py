import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Input, Layer
from tensorflow.keras.models import Model


class SimWalk(Layer):
    def __init__(self, b=None, b_init=.5, ** kwargs):
        super().__init__(**kwargs)
        self.b = b
        self.b_init = b_init

    def build(self, _):
        if self.b is None:
            self.b = tf.Variable(self.b_init)
        self.a = self.add_weight(shape=[1], name="a", initializer="ones")
        self.bias = self.add_weight(
            shape=[1], name="bias", initializer="ones")
        self.power = self.add_weight(
            shape=[1], name="power", initializer="ones")

    def call(self, inp):
        b = self.b

        # compute distances
        delta = inp[:, 1:] - inp[:, :-1]
        dist = tf.reduce_sum(delta**2, axis=-1)

        # avoid exploding values
        dist = tf.clip_by_value(dist, 1e-9, float("inf"))**(.5*b)

        # return tf.nn.sigmoid(-self.a * dist + self.bias)
        dist = tf.clip_by_value(self.a * dist + self.bias, 0, float("inf"))
        return 1 - tf.nn.tanh(dist)


def reduce(
    G, n_components=2, walk_len=40, n_walks=40, corruption=.5,
    batch_size=200, epochs=10, init="spectral", l2=1e-3, b=None,
    verbose=0
):
    import networkx as nx
    import numpy as np
    from tensorflow.keras import regularizers
    from walker import corrupt, random_walks, random_walks_with_weights

    # X_train = random_walks(G, walk_len=walk_len,
    #                        n_walks=n_walks, verbose=False)
    # y_train = corrupt(G, X_train, corruption, verbose=False)
    X_train, y_train = random_walks_with_weights(
        G, walk_len=walk_len, n_walks=n_walks, verbose=False)
    bypass = corrupt(G, X_train, corruption, verbose=False)

    y_train **= .005
    y_train *= bypass

    if init == "spectral":
        weights = nx.spectral_layout(G, dim=n_components)
        weights = [np.array([weights[i] for i in G.nodes])]
    else:
        weights = None

    walk_len = X_train.shape[1]

    inp = Input(shape=(walk_len,))
    x = Embedding(len(G.nodes), n_components,
                  mask_zero=False, weights=weights,
                  embeddings_regularizer=regularizers.l2(l2))(inp)
    x = SimWalk(b=b)(x)

    # early stopping
    callback = EarlyStopping(monitor="loss", min_delta=1e-4, patience=2)
    model = Model(inp, x)
    model.compile("nadam", "mse",
                  metrics=["mse"])
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose,
              callbacks=[callback])

    X = model.layers[1].get_weights()[0]
    return X
