"""
Core fucntions on tensors in Block Tensor-Train (BTT) format
"""

from typing import Optional, Union

import numpy as np
import tensorly as tl
from tensorly.tt_tensor import TTTensor


class BTTTensor(TTTensor):
    """
    A BTTTensor represents an (N+1)th order tensor of size I0 x I1 x ... x I[N-1] x K
    by a list of N factors, G0, G1, ..., G[N-1].

    All factors are 3rd order tensors.
    Only one factor, Gm, is of shape Rm x Im*K x R[m+1], and
    the rest, Gn, are of shape Rn x In x R[n+1].
    BTT-block_mode, m, is normally set at m = N-1.
    BTT-rank is a tuple of R0, R1, ..., RN, and
    BTT-shape is a tuple of I0, I1, xxx, I[N-1], K.
    """

    def __init__(
        self,
        factors: Union[float, int, list[np.ndarray]],
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
            shape_next = [f.shape[1] for f in factors[self.block_mode + 1, n_factors]]
            self.shape = tuple([*shape_prev, shape_curr, *shape_next, block_size])

    def __repr__(self):
        message = f"rank-{self.rank} mode-{self.block_mode} btt tensor of shape {self.shape} "
        return message

    def to_tensor(self):
        return btt_to_tensor(self)

    def to_unfolding(self, mode):
        return btt_to_unfolded(self, mode)

    def to_vec(self):
        return btt_to_mat(self)


def btt_to_tensor(btt_tensor: BTTTensor) -> np.ndarray:
    """Returns the full tensor whose BTT decomposition is given by 'factors'

    Re-assembles 'factors', which represent a tensor in BTT format
    into the corresponding full tensor

    Parameters
    ----------
    btt_tensor : BTTTensor

    Returns
    -------
    output_tensor : ndarray
        (N+1)th order tensor of size I0 x I1 x ... x I[N-1] x K,
        whose BTT cores were given by 'factors'.
    """
    if isinstance(btt_tensor.factors, (float, int)):  # 0-order tensor
        return btt_tensor.factors

    full_tensor = tl.ones((1, 1))
    for factor in btt_tensor:
        rank_prev, rank_next = factor.shape[0], factor.shape[-1]
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
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


def btt_to_unfolded(btt_tensor: BTTTensor, mode: int = 0):
    """Returns the unfolding matrix of a tensor given in BTT (or Tensor-Train) format

    Reassembles a full tensor from 'factors' and returns its unfolding matrix
    with mode given by 'mode' starting at 0.

    Parameters
    ----------
    btt_tensor: BTTTensor
        BTTTensor format with N factors representing an (N+1)th order tensor.
    mode: int
        unfolding matrix to be computed along this mode, having values from 0, 1, ..., N.

    Returns
    -------
    2-D array
        unfolding matrix at mode given by 'mode'
    """
    return tl.unfold(btt_to_tensor(btt_tensor), mode)


def btt_to_mat(btt_tensor: BTTTensor):
    """Returns the tensor defined by its BTT format into
       its matrix format of shape I x K with I=I1*I2*...*IN

    Parameters
    ----------
    btt_tensor: BTTTensor

    Returns
    -------
    2-D array
        matricization of BTT tensor into shape I x K with I=I1*I2*...*IN
    """
    return tl.reshape(btt_to_tensor(btt_tensor), (-1, btt_tensor.shape[-1]))
