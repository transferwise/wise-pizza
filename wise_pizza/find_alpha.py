import numpy as np
import pandas as pd
from scipy.sparse import vstack, csr_array, issparse
from scipy.linalg import svd, expm

from wise_pizza.solver import solve_lasso, solve_lp


def find_alpha(
    X,
    y_,
    max_nonzeros=None,
    min_nonzeros=None,
    verbose=0,
    max_iter: int = 100,
    solver: str = "lp",
    use_proj=None,
    constrain_signs=True,
    adding_up_regularizer=False,
):
    """
    Find alpha for optimal number of segments
    @param X: Train data
    @param y_: Target
    @param max_nonzeros: Maximum number of features to find
    @param min_nonzeros: Minimum number of features to find
    @param verbose: If set to a truish value, lots of debug info is printed to console
    @param max_iter: Maximum number of iterations to find segments
    @param solver: If this equals to "lp" uses the LP solver, else uses the (recommended) Lasso solver
    @param use_proj: For using projections
    @param constrain_signs: To constrain signs
    @param adding_up_regularizer: Force the contributions of detected segments to add up to the total difference
    ("other"=0). Experimental, use at your own risk
    @return: Fitted model and nonzeros
    """
    solve = solve_lp if solver == "lp" else solve_lasso
    if use_proj is None:
        use_proj = solve == solve_lp

    min_nonzeros, max_nonzeros = clean_up_min_max(min_nonzeros, max_nonzeros)

    if isinstance(X, pd.DataFrame):
        X = X.values

    y_ = np.array(y_).astype(X.dtype)
    if adding_up_regularizer:
        # add an additional row to the problem, to make the averages add up
        y_ = np.concatenate([y_, y_.sum(keepdims=True) * float(adding_up_regularizer)])
        if issparse(X):
            X = csr_array(vstack([X, X.sum(axis=0) * float(adding_up_regularizer)]))
        else:
            X = np.concatenate(
                [X, X.sum(axis=0, keepdims=True) * float(adding_up_regularizer)]
            )

    sparse_proj = True

    if sparse_proj:
        # Let's collapse the dimension of X and y by projecting both onto X
        # X = U @ S @ Vh so U.shape[0] == X.shape[0] == len(y)
        # Then X.T @ X = Vh.T.@S@S@Vh, and
        # for any matrix M, U.T @ M = Vh.T @ inv(S) @ X.T @ M
        # It's done in this way so the only huge matrix is X,
        # and we never have to create a matrix with one of the dimensions=len(X)

        XtX = (X.T.dot(X)).todense()
        _, s2, Vh = svd(XtX)
        s = np.sqrt(s2)
        eps = 1e-10 * s.max()
        s[s < eps] = eps
        Sinv = np.diag(1 / s)
        H = Sinv.dot(Vh)

        small_mat = H.dot(XtX)
        small_y = H.dot(X.T.dot(y_))
    else:
        # All this is just games on low-dimensional matrices
        # in order to diagnose why Lasso is not invariant to
        # simulateneous rotations of X and y_
        # let's collapse X and y onto the subspace spanned by X
        Xd = X.todense() if issparse(X) else X
        U, _, _ = svd(Xd, full_matrices=False)
        # let's just do a random rotation, closeness to np.eye parametrized by eps
        # eps = 1.0
        # randmat = np.random.randn(len(y_), len(y_))
        # U = expm(eps * (randmat.T - randmat))

        print("Biggest U deviation from eye:", np.max(np.abs(U - np.eye(len(U)))))
        maxerr = np.max(np.abs(U.T.dot(U) - np.eye(len(U))))
        # assert maxerr < 1e-5
        print("Biggest U.T@U deviation from eye:", maxerr)

        # # If we use a degenerate U, it all works
        # U = np.eye(len(U))

        # y: n x 1
        # X: n x m
        # a: m x 1
        #
        # err = y - X @ a
        # loss = err.T @ err + c * sum(abs(a))
        #
        # Xnew = U @ X
        # ynew = U @ y
        # err_new = U @ err
        # loss_new = loss

        small_mat = U.T.dot(Xd)
        small_y = U.T.dot(y_)
    # use for testing
    # assert np.max(np.abs(X.T.dot(X) - small_mat.T.dot(small_mat))) < 1e-2
    # assert (
    #     np.max(np.abs(X.T.dot(y_) - small_mat.T.dot(small_y)))
    #     / np.max(np.abs(small_mat.T.dot(small_y)))
    #     < 1e-5
    # )

    if use_proj:
        mat = small_mat
        y = small_y
    else:
        mat = X
        y = y_

    alpha = 2 * max(y)
    nonzeros = []
    iter = 0

    def print_errors(a: np.ndarray):
        err1 = X.dot(a) - y_
        err2 = np.array(small_mat.dot(a) - small_y)
        yerr = (y_ * y_).sum() - (small_y * small_y).sum()
        sqerr1 = np.sqrt((err1 * err1).sum() - yerr)
        sqerr2 = np.sqrt((err2 * err2).sum())
        d = (err1 * err1).sum() - (err2 * err2).sum() - yerr
        if verbose is not None:
            print("errors", sqerr1, sqerr2)

    if verbose:
        print_errors(np.zeros(X.shape[1]))

    while len(nonzeros) < min_nonzeros:
        alpha /= 2
        reg = solve(
            mat,
            y,
            alpha,
            constrain_signs=constrain_signs,
            verbose=verbose,
            drop_last_row=adding_up_regularizer,
        )
        if verbose:
            print_errors(reg.coef_)
        nonzeros = np.nonzero(reg.coef_)[0]
        iter += 1
        if iter > max_iter:
            break

    min_alpha = alpha
    while len(nonzeros) > max_nonzeros:
        alpha *= 2
        reg = solve(
            mat,
            y,
            alpha,
            constrain_signs=constrain_signs,
            verbose=verbose,
            drop_last_row=adding_up_regularizer,
        )
        if verbose:
            print_errors(reg.coef_)
        nonzeros = np.nonzero(reg.coef_)[0]
        iter += 1
        if iter > max_iter:
            break

    max_alpha = alpha

    # assert iter <= max_iter, "Alpha search failed - max_iter exceeded"
    # TODO: do the second part only on the first set of nonzeros
    alpha_lo = min_alpha
    alpha_hi = max_alpha
    while len(nonzeros) > max_nonzeros or len(nonzeros) < min_nonzeros:
        alpha = (alpha_hi + alpha_lo) / 2
        reg = solve(
            mat,
            y,
            alpha,
            constrain_signs=constrain_signs,
            verbose=verbose,
            drop_last_row=adding_up_regularizer,
        )
        if verbose:
            print_errors(reg.coef_)
        nonzeros = np.nonzero(reg.coef_)[0]
        iter += 1
        if iter > max_iter:
            break
        # assert iter <= max_iter, "Alpha search failed - max_iter exceeded"

        if len(nonzeros) < min_nonzeros:
            alpha_hi = alpha
        elif len(nonzeros) > max_nonzeros:
            alpha_lo = alpha
    if verbose:
        print("alpha=", alpha, nonzeros)
    X_filt = mat[:, nonzeros]
    reg = solve(
        X_filt,
        y,
        1e-3 * alpha,
        constrain_signs=constrain_signs,
        verbose=verbose,
        drop_last_row=adding_up_regularizer,
    )
    return reg, nonzeros


def clean_up_min_max(min_nonzeros: int = None, max_nonzeros: int = None):
    assert min_nonzeros is not None or max_nonzeros is not None
    if max_nonzeros is None:
        if min_nonzeros is None:
            max_nonzeros = 5
            min_nonzeros = 5
        else:
            max_nonzeros = min_nonzeros
    else:
        if min_nonzeros is None:
            min_nonzeros = max_nonzeros

    assert min_nonzeros <= max_nonzeros
    return min_nonzeros, max_nonzeros
