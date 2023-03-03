import torch

from metaseq.quantization.quantization_utils import (
    _QUANTIZATION_FUNCTIONS,
    initialize_quantization_state,
)


def main():
    torch.manual_seed(0)
    weight = torch.randn(10, 10).to(torch.float16)

    initialize_quantization_state()

    int8_weight, scale = _QUANTIZATION_FUNCTIONS["quantization_fn"](
        weight, bit_width=8
    )

    new_weight = _QUANTIZATION_FUNCTIONS["extraction_fn"](
        int8_weight.cuda(),
        scale.cuda(),
        bit_width=8
    ).cpu()

    assert torch.allclose(weight, new_weight, atol=1.2e-1)
    assert torch.allclose(weight.sum(), new_weight.sum(), atol=1e-1)


if __name__ == "__main__":
    main()
