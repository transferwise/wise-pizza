from sklearn.linear_model import Lasso
import numpy as np
import copy
from scipy.sparse import issparse
from scipy.optimize import linprog


def solve_lasso(
    X,
    y,
    alpha,
    constrain_signs=False,
    verbose=None,
    drop_last_row=True,
):
    """
    Lasso-based finder of unusual segments
    @param X: Matrix describing the segments
    @param y: target vector
    @param alpha: regularizer weight
    @param constrain_signs: Whether to constrain weights of segments to have the same
    sign as naive segment averages
    @param verbose: whether to print to output
    @param drop_last_row: Should be set to True if the last row of X
    is used to coerce adding-up of segment impacts to the total y average
    @return: Object returned by Lasso or LinearRegression
    """
    if issparse(X):
        X = X.todense()

    if alpha <= 0:
        alpha = 1e-6

    lasso_args = {
        "max_iter": int(1e5),
        "fit_intercept": False,
        "selection": "random",
        "positive": constrain_signs,  # forces the coefficients to be positive if True,
        "random_state": 42,
    }

    if constrain_signs:
        # make sure the segment coeffs have the same sign as the naive segment impact
        # last row of X is totals normalization,
        X = np.copy(X)
        if drop_last_row:
            Xy = X[:-1].T.dot(y[:-1])
        else:
            Xy = X.T.dot(y)
        neg_idxs = np.where(Xy < 0)[0]  # finding columns with negative
        X[:, neg_idxs] *= -1  # flip signs of X with negative averages

    lasso = Lasso(alpha=alpha, **lasso_args)
    lasso.fit(X, y)
    if constrain_signs:
        lasso.coef_[neg_idxs] *= -1  # flip the signs of the corresponding coefficients
    if verbose:
        nonzeros = np.nonzero(lasso.coef_)[0]
        print(alpha, nonzeros, lasso.coef_[nonzeros])
    return lasso


def solve_lp(X, y, alpha, constrain_signs=False, verbose=None, drop_last_row=True):
    """
    LP-based finder of unusual segments
    @param X: Matrix describing the segments
    @param y: target vector
    @param alpha: regularizer weight
    @param constrain_signs: Whether to constrain weights of segments to have the same
    sign as naive segment averages
    @param verbose: whether to print to output
    @param drop_last_row: Should be set to True if the last row of X
    is used to coerce adding-up of segment impacts to the total y average
    @return: Object returned by linprog, with coef_ and intercept_ attached for convenience
    """
    # TODO: try neater handling of sparse matrices
    if issparse(X):
        X = X.todense()

    """
    The math is as follows: 
    What we want is 
    | X @ a - y |_1 + \alpha | a |_1 -> min
    We use the usual trick to convert 1-norm into an LP problem,
    a := ap - an, 
    X@(ap - an) = sp - sn, 
    so x = [ap, an, sp, sn], x >=0
    and the objective is then 
    sp + sn + \alpha ( ap + an) -> min`
    """

    X_ = copy.copy(X)
    m = X.shape[1]

    c = np.ones(2 * m + 2 * len(y))
    A_eq = np.concatenate([X_, -X_, -np.eye(len(y)), np.eye(len(y))], axis=1)
    bounds = [(0.0, None) for i in range(len(c))]

    if constrain_signs:
        # make sure the segment coeffs have the same sign as the naive segment impact
        # last row of X is totals normalization,
        if drop_last_row:
            Xys = np.sign(X[:-1].T.dot(y[:-1]))
        else:
            Xys = np.sign(X.T.dot(y))

        Xys = np.array(Xys).reshape(-1)
        for i in range(m):
            if Xys[i] > 0:
                bounds[m + i] = (0, 0)
            else:
                bounds[i] = (0, 0)

    c[: 2 * m] *= alpha

    assert len(c) == A_eq.shape[1]
    x = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds)

    # for compatibility with the sklearn.linear models
    x.coef_ = x.x[:m] - x.x[m : 2 * m]  # * Xys
    x.intercept_ = x.x[2 * m]
    if verbose:
        nonzeros = np.nonzero(x.coef_)[0]
        print(alpha, nonzeros, x.coef_[nonzeros], "status:", x.status)
    return x
