import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Embedding, Input, Layer
from tensorflow.keras.models import Model


class SimWalk(Layer):
    def __init__(self, b=None, b_init=.5, ** kwargs):
        super().__init__(**kwargs)
        self.b = b
        self.b_init = b_init

    def build(self, _):
        if self.b is None:
            self.b = tf.Variable(self.b_init)
        self.a = tf.Variable(0.)

    def call(self, inp):
        b = self.b
        a = tf.math.log(tf.exp(1 + self.a))

        # compute distances
        delta = inp[:, 1:] - inp[:, :-1]
        dist = tf.reduce_sum(delta**2, axis=-1)
        # avoid exploding values
        dist = tf.clip_by_value(dist, 1e-9, float("inf"))**(.5*b)

        return 1 - tf.nn.tanh(a*dist)


def reduce(
    G, n_components=2, walk_len=50, n_walks=50, corruption=.5,
    batch_size=200, epochs=10, init="spectral", l2=1e-3, b=None
):
    import networkx as nx
    import numpy as np
    from tensorflow.keras import regularizers
    from walker import corrupt, random_walks

    X_train = random_walks(G, walk_len=walk_len,
                           n_walks=n_walks, verbose=False)
    y_train = corrupt(G, X_train, corruption, verbose=False)

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
    x = BatchNormalization()(x)
    x = SimWalk(b=b)(x)

    model = Model(inp, x)
    model.compile("rmsprop", "binary_crossentropy",
                  metrics=["binary_accuracy"])
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=0)

    X = model.layers[1].get_weights()[0]
    return X
