import numpy as np


def entropy(labels):
    """ Computes entropy of 0-1 vector. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)


def get_scoring_function(score):
    if score == "modularity":
        def scoring(_, __, Q, labels, distances, sigma_count):
            return Q
    elif score == "davies_bouldin":
        from sklearn.metrics import davies_bouldin_score

        def scoring(X, y, _, labels, distances, sigma_count):
            return -davies_bouldin_score(X, y)
    elif score == "silhouette":
        from sklearn.metrics import silhouette_score

        def scoring(X, y, _, labels, distances, sigma_count):
            return silhouette_score(X, y)
    elif score == "calinski_harabasz":
        from sklearn.metrics import calinski_harabasz_score

        def scoring(X, y, _, labels, distances, sigma_count):
            return calinski_harabasz_score(X, y)
    elif score == "combine":
        from sklearn.metrics import (calinski_harabasz_score,
                                     davies_bouldin_score)

        def scoring(X, y, Q, labels, distances, sigma_count):
            # return 2/(1/(calinski_harabasz_score(X, y)+1e-6) + 1/(Q+1e-6))
            a = max(calinski_harabasz_score(X, y), 1e-6)
            b = max(-davies_bouldin_score(X, y), 1e-6)
            Q = max(Q, 1e-6)
            return 3/(1/a + 1/b + 1/Q)

    elif score == "isolation":
        import numpy as np

        def scoring(X, y, Q, labels, distances, sigma_count):
            import math
            n_coms = len(set(y))
            if n_coms == 1:  # only one cluster
                return -float("inf")

            agreement = 0
            neighbors_com = y[labels]
            sigma_count = np.array(sigma_count)
            neighbors_weights = sigma_count[labels]
            n, k = X.shape

            for val, row, weights, dists in zip(
                    y, neighbors_com, neighbors_weights, distances):

                same_com = row == val
                max_value = weights.sum()
                res = np.sum(weights * same_com) / max_value
                agreement += res
            agreement /= n
            agreement = max(agreement, 1e-6)
            Q = max(Q, 1e-6)
            # print(agreement, Q, n_coms)
            return Q + agreement

            # for i, neighbors in enumerate(labels):
            #     neighbors_com = y[]
    return scoring
