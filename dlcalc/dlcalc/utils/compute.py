"""
https://arxiv.org/pdf/2401.14489
https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
"""

from .math import ceil_divide, product


def compute_gemm_flops(n_tokens: int, weight_shape: tuple[int, ...]) -> float:
    """compute the number of FLOPs in a linear layer. Given by 2MNK."""
    return 2 * n_tokens * product(*weight_shape)


def compute_gemm_n_tiles(
    n_tokens: int, weight_shape: tuple[int, ...], tile_shape: tuple[int, int, int]
) -> int:
    """get the total number of tiles for a GEMM. GEMMs can be tiled:
    * along output dimension M
    * along output dimension N
    * along reduction dimension K (requires reduction)
    """
    gemm_m = n_tokens
    gemm_k, gemm_n = weight_shape

    # tile_m, tile_n = (256, 128) if gemm_m > gemm_n else (128, 256)
    tile_m, tile_n = 128, 128
    tile_k = gemm_k

    return product(
        ceil_divide(gemm_m, tile_m),
        ceil_divide(gemm_n, tile_n),
        ceil_divide(gemm_k, tile_k),
    )
