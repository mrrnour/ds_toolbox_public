
def pca_ortho_rotation(lam,
                   method  = 'varimax',
                   gamma   = None,
                   eps     = 1e-6,
                   itermax = 100
                   ):
    """
    ##TODO: document it 
    ## A VARIMAX rotation is a change of coordinates used in principal component analysis1 (PCA) that maximizes the sum of the variances of the squared loadings
    ## https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
    Return orthogal rotation matrix
    TODO: - other types beyond 
    """
    if gamma == None:
        if (method == 'varimax'):
            gamma = 1.0
        if (method == 'quartimax'):
            gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0

    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp     = np.diag(np.sum(lam_rot ** 2, axis = 0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R       = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new

    return R

def pca_important_features(transformed_features, components_, columns):
    import math
        ##TODO: check it and make a function
    ###http://benalexkeen.com/principle-component-analysis-in-python/    
    """
    This function will return the most "important" 
    features so we can determine which have the most
    effect on multi-dimensional scaling
    """
    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    # important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    important_features = pd.Series(important_features)
    important_features = important_features.sort_values(ascending=[False])
    return important_features

def pc_draw_vectors(transformed_features, components_, columns):
    """
    This funtion will project your *original* features
    onto your principal component feature-space, so that you can
    visualize how "important" each one was in the
    multi-dimensional scaling

    https://benalexkeen.com/principle-component-analysis-in-python/
    """
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ax = plt.axes()

    for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax

def split_multiLabel_data__index(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split

    Parameters
    ----------
    test_size : float, [0,1]
        the proportion of the dataset to include in the test split, the rest will be put in the train set

    random_state : None | int | np.random.RandomState
        the random state seed (optional)

    Returns
    -------
    X_train, y_train, X_test, y_test
        stratified division into train/test split
    """
    # from skmultilearn.model_selection import IterativeStratification
    # stratifier = IterativeStratification(
    #     n_splits=2,
    #     order=2,
    #     sample_distribution_per_fold=[test_size, 1.0 - test_size],
    #     random_state=random_state,
    #     # shuffle=True if random_state is not None else False
    # )

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    stratifier = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)    
    train_indexes, test_indexes = next(stratifier.split(X, y))

    return train_indexes,test_indexes
