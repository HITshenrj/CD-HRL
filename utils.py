import numpy as np
import networkx as nx
from scipy.stats import norm


def simulate_random_dag(d: int,
                        degree: float,
                        graph_type: str,
                        w_range: tuple = (0.5, 2.0)) -> nx.DiGraph:
    """Simulate random DAG.
    Args:
        d: variable number
        degree: DAG degree
        graph_type: {'erdos-renyi'}
        w_range: weight range

    Returns:
        G: random DAG
    """
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    else:
        raise ValueError('unknown graph type')
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G


def simulate_sem(G: nx.DiGraph,
                 n: int,
                 sem_type: str,
                 linear_type: str) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss}
        linear_type:{linear,linear_sin,linear_tanh,linear_cos,linear_sigmoid}

    Returns:
        X: [n,d] sample matrix
    """
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    if sem_type == 'linear-gauss':
        X = np.random.randn(n, d)
    else:
        raise ValueError('unknown sem type')
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    # X=aX+bf(X)+delta
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        eta = X[:, parents].dot(W[parents, j])
        if linear_type == 'linear':
            X[:, j] += eta
        elif linear_type == 'sin':
            X[:, j] += 2.*np.sin(eta) + eta
        elif linear_type == 'linear_tanh':
            X[:, j] += 2.*np.tanh(eta) + eta
        elif linear_type == 'linear_cos':
            X[:, j] += 2.*np.cos(eta) + eta
        elif linear_type == 'linear_sigmoid':
            X[:, j] += 2.*sigmoid(eta) + eta
    return X


def count_accuracy(B_true: np.ndarray,
                   B: np.ndarray) -> tuple:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        G_true: ground truth graph
        G: predicted graph

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    d = B.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size


class CondIndepParCorr():
    def __init__(self,
                 data: np.ndarray,
                 n: int) -> None:
        """Compute Fisher-Z.

        Args:
            data: sample data
            n: data sample size
        """
        super().__init__()
        self.correlation_matrix = np.corrcoef(data)
        self.num_records = n

    def calc_statistic(self,
                       x: int,
                       y: int,
                       zz: tuple) -> float:
        """Compute Fisher-Z.

        Args:
            x: Fisher-Z x
            y: Fisher-Z y
            zz: conditional set zz

        Returns:
            Fisher-Z(x,y|zz)
        """
        corr_coef = self.correlation_matrix
        if len(zz) == 0:
            par_corr = corr_coef[x, y]
        elif len(zz) == 1:
            z = zz[0]
            par_corr = (
                (corr_coef[x, y] - corr_coef[x, z]*corr_coef[y, z]) /
                np.sqrt((1-np.power(corr_coef[x, z], 2))
                        * (1-np.power(corr_coef[y, z], 2)))
            )
        else:  # zz contains 2 or more variables
            all_var_idx = (x, y) + zz
            corr_coef_subset = corr_coef[np.ix_(all_var_idx, all_var_idx)]
            # consider using pinv instead of inv
            inv_corr_coef = -np.linalg.pinv(corr_coef_subset)
            par_corr = inv_corr_coef[0, 1] / \
                np.sqrt(abs(inv_corr_coef[0, 0]*inv_corr_coef[1, 1]))

        # log( (1+par_corr)/(1-par_corr) )
        z = np.log1p(2*par_corr / (1-par_corr))
        val_for_cdf = abs(
            np.sqrt(self.num_records - len(zz) - 3) *
            0.5 * z
        )
        statistic = 2*(1-norm.cdf(val_for_cdf))
        return statistic
