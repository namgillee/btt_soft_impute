"""
Test functions for btt_tensor.py
"""

import tensorly as tl

from btt_soft_impute import btt_tensor as btt


def test_btttensor():
    """Test BTTTensor"""
    factors = [tl.tensor([[[1.0], [2.0]]]), tl.tensor([[[1.0], [1.0], [1.0], [2.0], [3.0], [4.0]]])]
    btt_tensor = btt.BTTTensor(factors, block_size=3)

    assert btt_tensor.block_mode == 1
    assert btt_tensor.shape == (2, 2, 3)
    assert btt_tensor.rank == (1, 1, 1)

    targ_tensor = tl.stack(
        [
            tl.tensor([[1.0, 2.0], [2.0, 4.0]]),
            tl.tensor([[1.0, 3.0], [2.0, 6.0]]),
            tl.tensor([[1.0, 4.0], [2.0, 8.0]]),
        ],
        axis=2,
    )

    assert tl.all(btt_tensor.to_tensor() == targ_tensor)
    assert tl.all(btt_tensor.to_unfolding(0) == tl.reshape(targ_tensor, (2, -1)))
    assert tl.all(
        btt_tensor.to_unfolding(1) == tl.reshape(tl.transpose(targ_tensor, (1, 0, 2)), (2, -1))
    )
    assert tl.all(
        btt_tensor.to_unfolding(2) == tl.reshape(tl.transpose(targ_tensor, (2, 0, 1)), (3, -1))
    )
    assert tl.all(btt_tensor.to_vec() == tl.reshape(targ_tensor, (-1,)))
    assert tl.all(btt_tensor.to_mat() == tl.reshape(targ_tensor, (-1, 3)))


def test_btt_dot():
    """Test btt_dot"""
    factors = [tl.tensor([[[1.0], [2.0]]]), tl.tensor([[[1.0], [1.0], [1.0], [2.0], [3.0], [4.0]]])]
    btt_tensor = btt.BTTTensor(factors, block_size=3)

    desired_result = tl.tensor([[25.0, 35.0, 45.0], [35.0, 50.0, 65.0], [45.0, 65.0, 85.0]])

    assert tl.all(btt.btt_dot(btt_tensor, btt_tensor) == desired_result)


def test_btt_rand():
    """Test btt_rand"""
    btt_tensor = btt.btt_rand((2, 2, 3), (1, 3, 1))

    assert btt_tensor.block_mode == 1
    assert btt_tensor.shape == (2, 2, 3)
    assert btt_tensor.rank == (1, 2, 1)
    assert tl.max(tl.abs(btt.btt_dot(btt_tensor, btt_tensor) - tl.eye(3))) < 1e-10
