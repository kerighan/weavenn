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
        import numpy as np

        def scoring(X, y, Q, labels, distances, sigma_count):
            l = np.unique(y)
            n_samples = X.shape[0]
            n_labels = len(l)
            if n_labels == 1:
                return -float("inf")

            extra_disp, intra_disp = 0.0, 0.0
            mean = np.mean(X, axis=0)
            for k in l:
                cluster_k = X[y == k]
                mean_k = np.mean(cluster_k, axis=0)
                extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
                intra_disp += np.sum((cluster_k - mean_k) ** 2)

            value = (
                1.0
                if intra_disp == 0.0
                else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
            )
            return value
    elif score == "mch":
        import numpy as np

        def scoring(X, y, Q, labels, distances, sigma_count):
            labels = np.unique(y)
            n_samples = X.shape[0]
            n_labels = len(labels)
            if n_labels == 1:
                return -float("inf")

            sigma_count = np.array(sigma_count)
            sigma_count_normalized = sigma_count / sigma_count.sum()
            extra_disp, intra_disp = 0.0, 0.0

            # mean = np.mean(X, axis=0)
            mean = np.sum(X * sigma_count_normalized[:, None], axis=0)
            for k in labels:
                cluster_k = X[y == k]
                if len(cluster_k) == 1:
                    mean_k = cluster_k.mean(axis=0)
                    sigma_k_sum = sigma_count[y == k].sum()
                else:
                    sigma_k = sigma_count[y == k]
                    sigma_k_sum = sigma_k.sum()
                    sigma_k /= sigma_k_sum
                    mean_k = np.sum(sigma_k[:, None] * cluster_k, axis=0)
                extra_disp += sigma_k_sum * np.sum((mean_k - mean) ** 2)
                intra_disp += np.sum((cluster_k - mean_k) ** 2)

            value = (
                1.0
                if intra_disp == 0.0
                else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
            )
            # return value * Q
            # print(value * Q)
            return value * Q
    elif score == "test2":
        import numpy as np

        def scoring(X, y, Q, labels, distances, sigma_count):
            l = np.unique(y)
            n_samples = X.shape[0]
            n_labels = len(l)
            if n_labels == 1:
                return -float("inf")

            sigma_count = np.array(sigma_count)
            extra_disp, intra_disp = 0.0, 0.0
            edge_disp = 0.0

            # mean = np.mean(X, axis=0)
            neighbors_com = y[labels]
            for k in l:
                labels_k = y == k
                cluster_k = X[labels_k]
                dist_k = distances[labels_k]
                if len(cluster_k) == 1:
                    sigma_k_sum = sigma_count[labels_k].sum()
                else:
                    sigma_k = sigma_count[labels_k]
                    sigma_k_sum = sigma_k.sum()
                    sigma_k /= sigma_k_sum

                same_com = neighbors_com[labels_k] == k

                intra_disp += np.sum(dist_k[same_com == True]**2)
                extra_disp += np.sum(dist_k[same_com == False]**2)

                # edge_disp += sigma_k_sum * np.sum(dist_k**2)

            value = (
                1.0
                if intra_disp == 0.0
                else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
            )
            return value * Q

    elif score == "isolation":
        import numpy as np

        def scoring(X, y, Q, labels, distances, sigma_count):
            labels = labels[:, :20]
            distances = distances[:, :20]

            n_coms = len(set(y))
            if n_coms == 1:  # only one cluster
                return -float("inf")

            agreement = 0
            neighbors_com = y[labels]
            sigma_count = np.array(sigma_count)
            neighbors_weights = sigma_count[labels]
            n, k = labels.shape

            # discount = np.array([(1/i) for i in range(1, k+1)])
            # discount = np.linspace(k, 1, k)
            # discount = np.linspace(1, .1, k)

            for val, row, weights, dists in zip(
                    y, neighbors_com, neighbors_weights, distances):
                # weights *= discount
                # weights **= 2
                same_com = row == val

                res = np.sum(weights * same_com) / weights.sum()
                # res = np.mean(same_com)
                agreement += res
            agreement /= n
            agreement = max(agreement, 1e-6)
            Q = max(Q, 1e-6)
            # return 2/(1/agreement + 1/Q)
            # return (agreement**(1/np.log(k))) * Q
            # return (agreement + (k - 1) * Q) / k
            return Q * agreement
    elif score == "isolation2":
        import numpy as np

        def scoring(X, y, Q, labels, distances, sigma_count):
            labels = labels[:, :20]
            distances = distances[:, :20]

            ys = np.unique(y)
            n_samples = X.shape[0]
            n_labels = len(ys)
            if n_labels == 1:
                return -float("inf")

            neighbors_com = y[labels]
            sigma_count = np.array(sigma_count)

            isolation = 0.
            total_sigma_k_sum = 0.
            for k in ys:
                labels_k = y == k
                cluster_k = X[labels_k]
                if len(cluster_k) == 1:
                    sigma_k_sum = sigma_count[labels_k].sum()
                else:
                    sigma_k = sigma_count[labels_k]
                    sigma_k_sum = sigma_k.sum()

                same_com = neighbors_com[labels_k] == k
                isolation += np.mean(same_com) * sigma_k_sum
                total_sigma_k_sum += sigma_k_sum
            isolation /= total_sigma_k_sum

            return Q * isolation

            # for i, neighbors in enumerate(labels):
            #     neighbors_com = y[]
    return scoring
