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

            H = 0
            for val, row, weights, dists in zip(
                    y, neighbors_com, neighbors_weights, distances):

                same_com = row == val

                H += entropy(row)

                max_value = weights.sum()
                res = np.sum(weights * same_com) / max_value

                agreement += res
            agreement /= n
            H /= n
            # harmonic = (10) / (1/Q + 9/agreement)
            # print(agreement, variance, Q)
            agreement = -H
            print(agreement)
            # print(agreement, variance, Q, n_coms)
            # agreement = agreement * math.log(1 + Q)
            # agreement = 100 * agreement + Q
            # print(agreement)
            # agreement *= Q
            # if agreement == 1:
            #     agreement = .97
            # agreement *= math.log(1 + n_coms)
            # print(agreement)
            # raise KeyError
            # agreement = agreement / variance
            return agreement

            # for i, neighbors in enumerate(labels):
            #     neighbors_com = y[]
    return scoring
