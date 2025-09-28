from cs336_basics.model.attention import MultiHeadAttention
import torch


def test_casual_mask():
    multihead = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(10, 20, 512)
    mask = multihead._create_mask(x.shape, 20, x.device, None)
    assert mask.shape == (10, 20, 20)
    tri = torch.tril(torch.ones(20, 20)).bool()
    assert (mask == tri).all()


def test_casual_mask_with_positions():
    multihead = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(10, 20, 512)
    positions = torch.arange(20).unsqueeze(0).repeat(10, 1)  # Shape: (10, 20)
    mask = multihead._create_mask(x.shape, 20, x.device, positions)
    assert mask.shape == (10, 20, 20)
    tri = torch.tril(torch.ones(20, 20)).bool()
    assert (mask == tri).all()


def test_casual_mask_with_positions_and_no_monotonicity():
    multihead = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(10, 5, 512)
    positions = torch.tensor([[0, 1, 2, 0, 1]] * 10)  # Shape: (10, 5)
    mask = multihead._create_mask(x.shape, 5, x.device, positions)
    assert mask.shape == (10, 5, 5)
    expected_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
        ]
    ).bool()
    assert (mask[0] == expected_mask).all()
