import math
import numpy as np

def ranking_matrix(D, solver='fast'):
    '''
    Get the ranking matrix from distance matrix

    Parameters
    ----------
    D - distance matrix
    solver - default 'fast'.

    Return
    ------
    R - ranking matrix
    '''

    D = np.array(D)

    if solver == 'fast':
        # Computing ranking_matrix using np.argsort twice on every row. By Nilrad
        R = [np.argsort(np.argsort(row)) for row in D]
    else:
        R = np.zeros(D.shape)
        m = len(R)

        for i in range(m):
            for j in range(m):
                Rij = 0
                for k in range(m):
                    if (D[i,k] < D[i,j]) or (math.isclose(D[i,k], D[i,j]) and k < j ):
                        Rij += 1
                R[i,j] = Rij

    return np.array(R, dtype = 'uint')

def coranking_matrix(R1, R2):
    '''
    Get co-ranking matrix

    Parameters
    ----------
    R1, R2 - two ranking matrices as input

    Return
    ------
    Q - coranking matrix
    '''


    R1 = np.array(R1)
    R2 = np.array(R2)
    assert R1.shape == R2.shape
    Q = np.zeros(R1.shape)
    m = len(Q)

    for i in range(m):
        for j in range(m):
            k = int(R1[i,j])
            l = int(R2[i,j])
            Q[k,l] += 1

    return Q

def slice_Q(Q_full):
    """
    Drop the 0-th row/col (self–neighbors).
    Returns Q of shape (m, m).
    """
    return Q_full[1:, 1:]


def compute_trustworthiness(Q_full: np.ndarray) -> np.ndarray:
    """
    T[k] for k = 1..m-1
    """
    Q = slice_Q(Q_full)
    m = Q.shape[0]
    T = np.zeros(m-1)
    for k in range(m-1):
        # 1 - normalized hard-k-intrusions. lower-left region. weighted by rank error (rank - k)
        Qs = Q[k:, :k]
        # a column vector of weights. weight = rank error = actual_rank - k
        W = np.arange(Qs.shape[0]).reshape(-1,1)
        denom = (k+1) * m * (m - k - 1)
        T[k] = 1 - np.sum(Qs * W) / denom
    return T


def compute_auc_T(T: np.ndarray) -> float:
    """
    AUC of the Trustworthiness (just its average over k).
    """
    return np.mean(T)


def compute_continuity(Q_full: np.ndarray) -> np.ndarray:
    """
    C[k] for k = 1..m-1
    """
    Q = _slice_Q(Q_full)
    m = Q.shape[0]
    C = np.zeros(m-1)
    for k in range(m-1):
        # 1 - normalized hard-k-extrusions. upper-right region
        Qs = Q[:k, k:]
        # a row vector of weights. weight = rank error = actual_rank - k
        W = np.arange(Qs.shape[1]).reshape(1, -1)
        denom = (k+1) * m * (m - 1 - k)
        C[k] = 1 - np.sum(Qs * W) / denom
    return C

def compute_auc_C(C: np.ndarray) -> float:
    """
    AUC of the Continuity (just its average over k).
    """
    return np.mean(C)


def compute_QNN(Q_full: np.ndarray) -> np.ndarray:
    """
    QNN[k] for k = 0..m-1  (0-th NN is the point itself)
    """
    Q = slice_Q(Q_full)
    m = Q.shape[0]
    QNN = np.zeros(m)
    for k in range(m):
        # Q[0,0] is always m. 0-th nearest neighbor is always the point itself. Exclude Q[0,0]
        QNN[k] = np.sum(Q[:k+1, :k+1]) / ((k+1) * m)
    return QNN


def compute_auc_QNN(QNN: np.ndarray) -> float:
    """
    AUC of the QNN curve (just its average over k).
    """
    return np.mean(QNN)


def compute_LCMC(QNN: np.ndarray) -> np.ndarray:
    """
    LCMC[k] = QNN[k] - (k+1)/(m-1), for k = 0..m-1
    """
    m = len(QNN)
    # note: QNN[0] = 1 by definition, and baseline is 1/(m-1)
    return QNN - np.arange(1, m+1)/(m-1)


def compute_kmax(LCMC: np.ndarray) -> int:
    """
    Index k at which LCMC is maximized.
    """
    return int(np.argmax(LCMC))


def compute_Qlocal(QNN: np.ndarray, kmax: int) -> float:
    """
    Average QNN over k = 0..kmax
    """
    return np.mean(QNN[:kmax+1])


def compute_Qglobal(QNN: np.ndarray, kmax: int) -> float:
    """
    Average QNN over k = kmax+1..m-2
    (we skip the last entry at k=m-1, which is always 1)
    """
    return np.mean(QNN[kmax : -1])