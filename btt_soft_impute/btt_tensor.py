"""
Core functions on tensors in Block Tensor-Train (BTT) format
"""

import logging
from typing import Optional, Union

import tensorly as tl
from numpy.typing import NDArray
from tensorly.tt_tensor import TTTensor

logger = logging.getLogger(__name__)


class BTTTensor(TTTensor):
    """
    A BTTTensor represents an (N+1)th order tensor of size I0 x I1 x ... x I[N-1] x K
    by a list of N factors, G0, G1, ..., G[N-1].

    All factors are 3rd order tensors.
    Only one factor, Gm, is of shape Rm x (Im * K) x R[m+1], and
    the rest, Gn, are of shape Rn x In x R[n+1].
    BTT-block_mode, m, is usually set at m = N-1 by default.
    BTT-rank is a tuple of R0, R1, ..., RN, with R0=RN=1, and
    BTT-shape is a tuple of I0, I1, ..., I[N-1], K.
    """

    def __init__(
        self,
        factors: Union[float, int, list[NDArray]],
        block_mode: Optional[int] = None,
        block_size: int = 1,
        inplace: bool = False,
    ):
        # Will raise an error if invalid TTTensor
        super().__init__(factors=factors, inplace=inplace)

        if isinstance(factors, (float, int)):
            # Tensor is a scalar
            self.block_mode = None
            self.shape = (*self.shape, 1)
        else:
            n_factors = len(factors)
            self.block_mode = block_mode if block_mode else n_factors - 1

            # Check for valid block size
            shape_curr = int(factors[self.block_mode].shape[1] / block_size)
            if shape_curr * block_size != factors[self.block_mode].shape[1]:
                raise ValueError(
                    f"The shape of BTT-core at block_mode {self.block_mode} is not"
                    f" a multiple of block_size {block_size}."
                )

            shape_prev = [f.shape[1] for f in factors[: self.block_mode]]
            shape_next = [f.shape[1] for f in factors[(self.block_mode + 1) : n_factors]]
            self.shape = tuple([*shape_prev, shape_curr, *shape_next, block_size])

    def __repr__(self):
        message = f"rank-{self.rank} mode-{self.block_mode} btt tensor of shape {self.shape} "
        return message

    def to_tensor(self) -> NDArray:
        return btt_to_tensor(self)

    def to_unfolding(self, mode: int = 0) -> NDArray:
        """Unfolding matrix of a tensor given in BTT format.

        Parameters
        ----------
        btt_tensor: BTTTensor
            BTTTensor format with N factors representing an (N+1)th order tensor
            with shape I0 x I1 x ... x I[N-1] x K.
        mode: int
            unfolding matrix to be computed along this mode, having values from 0, 1, ..., N.

        Returns
        -------
        2-D tensorly.tensor
            unfolding matrix at mode given by 'mode'
        """
        return tl.unfold(self.to_tensor(), mode)

    def to_vec(self) -> NDArray:
        return tl.tensor_to_vec(self.to_tensor())

    def to_mat(self) -> NDArray:
        """Long-and-thin matrix defined by the BTT format.

        Parameters
        ----------
        btt_tensor: BTTTensor

        Returns
        -------
        2-D tensorly.tensor
            The returned matrix has the shape I x K with I=I0*I1*...*I[N-1].
        """
        return tl.reshape(self.to_tensor(), (-1, self.shape[-1]))


def btt_to_tensor(btt_tensor: BTTTensor) -> NDArray:
    """Returns the full tensor whose BTT decomposition is given by 'factors'

    Re-assembles 'factors', which represent a tensor in BTT format
    into the corresponding full tensor

    Parameters
    ----------
    btt_tensor : BTTTensor

    Returns
    -------
    output_tensor : tensorly.tensor
        (N+1)th order tensor of size I0 x I1 x ... x I[N-1] x K,
        whose BTT cores were given by 'factors'.
    """
    if isinstance(btt_tensor.factors, (float, int)):  # 0-order tensor
        return btt_tensor.factors

    full_tensor = tl.ones((1, 1))
    for factor in btt_tensor:
        rank_prev, rank_next = factor.shape[0], factor.shape[-1]
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.matmul(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    # Shift the block mode to the last mode
    block_mode = btt_tensor.block_mode
    block_size = btt_tensor.shape[-1]
    if block_mode < len(btt_tensor) - 1:
        shape_prev = btt_tensor.shape[: (block_mode + 1)]
        shape_next = btt_tensor.shape[(block_mode + 1) : -1]
        full_tensor = tl.reshape(
            full_tensor, [tl.prod(shape_prev), block_size, tl.prod(shape_next)]
        )
        full_tensor = tl.transpose(full_tensor, (0, 2, 1))

    return tl.reshape(full_tensor, btt_tensor.shape)


def btt_qr(btt_tensor: BTTTensor) -> tuple[BTTTensor, NDArray]:
    """Compute the QR factorization of the matrix representation of a BTT tensor.

    The matrix representation of a BTT tensor is considered of shape I x K with
    I = I0 * ... * I[N-1]. By QR factorization, the q is a BTT tensor and r is an
    upper-triangular matrix.

    Parameters
    ----------
    btt_tensor : BTTTensor

    Returns
    -------
    btt_tensor : BTTTensor
        The QR Q matrix in BTT format, having the same shape and ranks as the input BTT tensor,
        with orthogonalized core tensors.
    crr: NDArray
        The QR R matrix.
    """
    block_mode = btt_tensor.block_mode
    block_size = btt_tensor.shape[-1]
    n_cores = len(btt_tensor)
    ranks = btt_tensor.rank
    shapes = btt_tensor.shape

    # Left-to-right orthogonalization
    n = 0
    while n < block_mode:
        cr = btt_tensor.factors[n]
        cr = tl.reshape(cr, (ranks[n] * shapes[n], ranks[n + 1]))

        # Compute QR decomposition
        cr, crr = tl.qr(cr)

        # Update TT-rank
        ranks[n + 1] = crr.shape[0]

        # Update current core
        cr = tl.reshape(cr, (ranks[n], shapes[n], ranks[n + 1]))
        btt_tensor.factors[n] = cr

        # Multiply crr to the next core
        cr2 = btt_tensor.factors[n + 1]
        sz2 = cr2.shape
        cr2 = tl.reshape(cr2, (sz2[0], -1))
        cr2 = tl.matmul(crr, cr2)
        cr2 = tl.reshape(cr2, (ranks[n + 1], *sz2[1:]))
        btt_tensor.factors[n + 1] = cr2

        n = n + 1

    # Right-to-left orthogonalization
    n = n_cores - 1
    while n > block_mode:
        cr = btt_tensor.factors[n]
        cr = tl.reshape(cr, (ranks[n], shapes[n] * ranks[n + 1]))
        cr = tl.transpose(cr)

        # Compute QR decomposition
        cr, crr = tl.qr(cr)

        # Update TT-rank
        ranks[n] = crr.shape[0]

        # Update current core
        cr = tl.transpose(cr)
        cr = tl.reshape(cr, (ranks[n], shapes[n], ranks[n + 1]))
        btt_tensor.factors[n] = cr

        # Multiply crr to the next core
        cr2 = btt_tensor.factors[n - 1]
        sz2 = cr2.shape
        cr2 = tl.reshape(cr2, (-1, sz2[-1]))
        cr2 = tl.matmul(cr2, tl.transpose(crr))
        cr2 = tl.reshape(cr2, (*sz2[:-1], ranks[n]))
        btt_tensor.factors[n - 1] = cr2

        n = n - 1

    # Orthogonalize block_mode'th TT-core
    cr = btt_tensor.factors[block_mode]
    cr = tl.reshape(cr, (ranks[block_mode] * shapes[block_mode], block_size, ranks[block_mode + 1]))
    cr = tl.transpose(cr, (0, 2, 1))
    cr = tl.reshape(
        cr, (ranks[block_mode] * shapes[block_mode] * ranks[block_mode + 1], block_size)
    )

    cr, crr = tl.qr(cr)

    # Update block_size
    if block_size != crr.shape[0]:
        logger.warning("Block size will be changed")
    block_size = crr.shape[0]

    # Update current core
    cr = tl.reshape(cr, (ranks[block_mode], shapes[block_mode], ranks[block_mode + 1], block_size))
    cr = tl.transpose(cr, (0, 1, 3, 2))
    cr = tl.reshape(cr, (ranks[block_mode], shapes[block_mode] * block_size, ranks[block_mode + 1]))
    btt_tensor.factors[block_mode] = cr

    # Update BTT tensor
    btt_tensor.shape[-1] = block_size
    btt_tensor.rank = ranks

    return btt_tensor, crr


def tt_dot_left(tt1: TTTensor, tt2: TTTensor, current_mode: int = 0) -> NDArray:
    """Partial dot product of two TT tensors, between core tensors on the left of current mode.

    It returns a matrix of size R1[n] x R2[n], where n denotes the current mode.

    Parameters
    ----------
    tt1 : TTTensor
    tt2 : TTTensor
    current_mode : int
        Integer among 0, ..., N-1

    Returns
    -------
    prod : NDArray
        Partial dot product of tt1 and tt2 core tensors on the left of current mode.
        Matrix of size R1[n] x R2[n]
    """
    n_cores = len(tt1)
    current_mode = min(n_cores, current_mode)

    shapes = tt1.shape
    if shapes[:current_mode] != tt2.shape[:current_mode]:
        logger.warning("Sizes of two input TT tensors must be equal but they are different.")
        return tl.ones(shape=(1, 1))

    ranks1 = tt1.rank
    ranks2 = tt2.rank
    if ranks1[0] != 1 or ranks2[0] != 1:
        logger.warning("The first TT rank must be 1 but it is not.")
        return tl.ones(shape=(1, 1))

    prod = tl.ones(shape=(1, 1))
    for n in tl.arange(current_mode):
        cr1 = tt1.factors[n]
        cr2 = tt2.factors[n]

        # Multiply tt2 core
        cr2 = tl.reshape(cr2, (ranks2[n], shapes[n] * ranks2[n + 1]))
        prod = tl.matmul(prod, cr2)  # size r1n, In * r2+
        prod = tl.reshape(prod, (ranks1[n] * shapes[n], ranks2[n + 1]))

        # Multiply tt1 core
        cr1 = tl.reshape(cr1, (ranks1[n] * shapes[n], ranks1[n + 1]))
        prod = tl.matmul(tl.transpose(tl.conj(cr1)), prod)  # size r1+, r2+

    return prod


def tt_dot_right(tt1: TTTensor, tt2: TTTensor, current_mode: int = 0) -> NDArray:
    """Partial dot product of two TT tensors, between core tensors on the right of current mode.

    It returns a matrix of size R1[n + 1] x R2[n + 1], where n denotes the current mode.

    Parameters
    ----------
    tt1 : TTTensor
    tt2 : TTTensor
    current_mode : int
        Integer among 0, ..., N-1

    Returns
    -------
    prod : NDArray
        Partial dot product of tt1 and tt2 core tensors on the right of current mode.
        Matrix of size R1[n + 1] x R2[n + 1]
    """
    current_mode = max(-1, current_mode)

    shapes = tt1.shape
    if shapes[(current_mode + 1) :] != tt2.shape[(current_mode + 1) :]:
        logger.warning("Sizes of two input TT tensors must be equal but they are different.")
        return tl.ones(shape=(1, 1))

    ranks1 = tt1.rank
    ranks2 = tt2.rank
    n_cores = len(tt1)
    if ranks1[n_cores] != 1 or ranks2[n_cores] != 1:
        logger.warning("The last TT rank must be 1 but it is not.")
        return tl.ones(shape=(1, 1))

    prod = tl.ones(shape=(1, 1))
    for n in tl.arange(n_cores - 1, current_mode, -1):
        cr1 = tt1.factors[n]
        cr2 = tt2.factors[n]

        # Multiply tt1 core
        cr1 = tl.reshape(cr1, (ranks1[n] * shapes[n], ranks1[n + 1]))
        prod = tl.matmul(tl.conj(cr1), prod)
        prod = tl.reshape(prod, (ranks1[n], shapes[n] * ranks2[n + 1]))

        # Multiply tt2 core
        cr2 = tl.reshape(cr2, (ranks2[n], shapes[n] * ranks2[n + 1]))
        prod = tl.matmul(prod, tl.transpose(cr2))  # size r1n, r2n

    return prod


# pylint: disable=too-many-locals
def btt_dot(tt1: BTTTensor, tt2: BTTTensor, do_qr: bool = False) -> NDArray:
    """Dot product of two BTT tensors

    It returns a matrix of size K(tt1) x K(tt2),
    where K(tt) denotes the block size of a BTT tensor tt.

    Parameters
    ----------
    tt1 : BTTTensor
    tt2 : BTTTensor
    do_qr : bool
        If True, run QR factorization on tt1 and tt2, respectively.

    Returns
    -------
    prod : NDArray
        Dot product of tt1 and tt2. The matrix size is K(tt1) x K(tt2).
    """
    shapes = tt1.shape
    if shapes[:-1] != tt2.shape[:-1]:
        logger.warning("Sizes of two input TT tensors must be equal but they are different.")
        return tl.tensor(0.0)

    ranks1 = tt1.rank
    ranks2 = tt2.rank

    if do_qr:
        tt1, qrr1 = btt_qr(tt1)
        tt2, qrr2 = btt_qr(tt2)

    min_mode = min(tt1.block_mode, tt2.block_mode)
    max_mode = max(tt1.block_mode, tt2.block_mode)
    prod = tt_dot_left(tt1, tt2, current_mode=min_mode)
    prod_right = tt_dot_right(tt1, tt2, current_mode=max_mode)

    # Partial dot product of cores between tt1 and tt2, within min_mode and max_mode
    k10 = 1
    k20 = 1
    for n in tl.arange(min_mode, max_mode + 1):
        cr1 = tt1.factors[n]
        cr2 = tt2.factors[n]

        k1n = tt1.shape[-1] if n == tt1.block_mode else 1
        k2n = tt2.shape[-1] if n == tt2.block_mode else 1

        # Multiply tt2 core
        cr2 = tl.reshape(cr2, (ranks2[n], shapes[n] * k2n * ranks2[n + 1]))
        prod = tl.matmul(prod, cr2)  # size k10 * k20 * r1n, In * k2n * r2+
        prod = tl.reshape(prod, (k10 * k20 * ranks1[n] * shapes[n], k2n * ranks2[n + 1]))
        prod = tl.transpose(prod)
        prod = tl.reshape(prod, (k2n * ranks2[n + 1] * k10 * k20, ranks1[n] * shapes[n]))

        # Multiply tt1 core
        cr1 = tl.reshape(cr1, (ranks1[n] * shapes[n], k1n * ranks1[n + 1]))
        prod = tl.matmul(prod, tl.conj(cr1))  # size k2n * r2+ * k10 * k20, k1n * r1+

        # Reshape for the next iteration
        if n not in (tt1.block_mode, tt2.block_mode):
            prod = tl.reshape(prod, (ranks2[n + 1], k10 * k20 * ranks1[n + 1]))
            prod = tl.transpose(prod)
        elif n == tt1.block_mode and n != tt2.block_mode:
            prod = tl.reshape(prod, (ranks2[n + 1], k10, k20, k1n, ranks1[n + 1]))
            prod = tl.transpose(prod, (1, 3, 2, 4, 0))
            prod = tl.reshape(prod, (k10 * k1n * k20 * ranks1[n + 1], ranks2[n + 1]))
        elif n != tt1.block_mode and n == tt2.block_mode:
            prod = tl.reshape(prod, (k2n, ranks2[n + 1], k10 * k20, ranks1[n + 1]))
            prod = tl.transpose(prod, (2, 0, 3, 1))
            prod = tl.reshape(prod, (k10 * k20 * k2n * ranks1[n + 1], ranks2[n + 1]))
        else:
            prod = tl.reshape(prod, (k2n, ranks2[n + 1], k10, k20, k1n, ranks1[n + 1]))
            prod = tl.transpose(prod, (2, 4, 3, 0, 5, 1))
            prod = tl.reshape(prod, (k10 * k1n * k20 * k2n * ranks1[n + 1], ranks2[n + 1]))

        k10 = k10 * k1n
        k20 = k20 * k2n

    # Multiply prod_right
    prod = tl.reshape(prod, (k10 * k20, ranks1[max_mode + 1] * ranks2[max_mode + 1]))
    prod = tl.matmul(prod, tl.reshape(prod_right, (-1, 1)))
    prod = tl.reshape(prod, (k10, k20))

    if do_qr:
        prod = tl.matmul(tl.transpose(qrr1), tl.matmul(prod, qrr2))

    return prod
