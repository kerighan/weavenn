def get_hnswlib_nns_function(metric):
    def get_hnswlib_nns(
        X, k,
        ef_construction=200,
        M=16
    ):
        import hnswlib

        n, dim = X.shape

        index = hnswlib.Index(space=metric, dim=dim)
        index.init_index(
            max_elements=n, ef_construction=2*k, M=100,
            random_seed=100)
        index.add_items(X, range(n))
        # index.set_ef(2*k)

        labels, distances = index.knn_query(X, k=k)
        return labels, distances
    return get_hnswlib_nns


def get_scann_nns_function(metric):
    def get_scann_nns(X, k):
        searcher = scann.scann_ops_pybind.builder(
            X, 10, "euclidean").tree(
                num_leaves=2000,
                num_leaves_to_search=100,
                training_sample_size=25000).score_ah(
                    2,
                    anisotropic_quantization_threshold=0.2).reorder(
                        100).build()
        labels, distances = searcher.search_batched(queries)
        return labels, distances
    return get_scann_nns


def get_ann_algorithm(ann_algorithm, metric):
    if ann_algorithm == "hnswlib":
        return get_hnswlib_nns_function(metric)
    elif ann_algorithm == "scann":
        return get_scann_nns_function(metric)
    else:
        raise ValueError(f"Algorithm {ann_algorithm} not found")
