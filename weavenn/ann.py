def get_hnswlib_nns_function(metric):
    def get_hnswlib_nns(
        X, k,
        ef_construction=200,
        M=16
    ):
        import hnswlib

        n, dim = X.shape

        index = hnswlib.Index(space=metric, dim=dim)
        index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
        index.add_items(X, range(n))
        index.set_ef(50)

        labels, distances = index.knn_query(X, k=k)
        return labels, distances
    return get_hnswlib_nns


def get_ann_algorithm(ann_algorithm, metric):
    if ann_algorithm == "hnswlib":
        return get_hnswlib_nns_function(metric)
    else:
        raise ValueError(f"Algorithm {ann_algorithm} not found")
