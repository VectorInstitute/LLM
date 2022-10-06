from metaseq.quantization.quantization_utils import (
    initialize_quantization_state,
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear,
    quantize_state_dict,
    sanity_check_quantized_model,
)


if __name__ == "__main__":
    import torch
    from metaseq.quantization.quantization_utils import _QUANTIZATION_FUNCTIONS
    # TODO (mchoi): Move this to a test file
    weight = torch.randn(10, 10).to(torch.float16)

    initialize_quantization_state()

    int8_weight, scale = _QUANTIZATION_FUNCTIONS["quantization_fn"](weight, bit_width=8)

    new_weight = _QUANTIZATION_FUNCTIONS["extraction_fn"](int8_weight.cuda(), scale.cuda(), bit_width=8).cpu()

    assert torch.allclose(weight, new_weight, atol=1e-2)
    assert torch.allclose(weight.sum(), new_weight.sum(), atol=1e-1)
