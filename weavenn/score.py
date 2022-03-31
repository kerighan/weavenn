def get_scoring_function(score):
    if score == "modularity":
        def scoring(_, __, Q):
            return Q
    elif score == "davies_bouldin":
        from sklearn.metrics import davies_bouldin_score

        def scoring(X, y, _):
            return -davies_bouldin_score(X, y)
    elif score == "silhouette":
        from sklearn.metrics import silhouette_score

        def scoring(X, y, _):
            return silhouette_score(X, y)
    elif score == "calinski_harabasz":
        from sklearn.metrics import calinski_harabasz_score

        def scoring(X, y, _):
            return calinski_harabasz_score(X, y)
    return scoring
