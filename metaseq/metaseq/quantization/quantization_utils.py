import ctypes
import functools
from functools import partial
import torch
import torch.nn as nn
import pkg_resources

from cpm_kernels.kernels.base import (
    LazyKernelCModule,
    KernelFunction,
    round_up,
)
from megatron.mpu.layers import ColumnParallelLinear, RowParallelLinear
from megatron.mpu.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)


RESOURCE_PACKAGE_NAME = __name__


# Global to hold quantization functions post kernel-init
_QUANTIZATION_FUNCTIONS = {}


class _QuantizationKernelManager:
    """Class which initializes kernels, and holds the kernel functions."""

    def __init__(
        self,
        filepath,
    ):
        if not filepath.endswith(".fatbin"):
            filepath += ".fatbin"

        if not pkg_resources.resource_exists(RESOURCE_PACKAGE_NAME, filepath):
            raise RuntimeError(
                "File `%s` not found in `%s`" % (filepath, RESOURCE_PACKAGE_NAME)
            )
        self.filename = filepath
        # Get the compiled kernels as bytes
        self.code = pkg_resources.resource_string(RESOURCE_PACKAGE_NAME, filepath)
        self._function_names = [
            "int4WeightCompression",
            "int4WeightExtractionFloat",
            "int4WeightExtractionHalf",
            "int8WeightExtractionFloat",
            "int8WeightExtractionHalf",
        ]
        # Create the cuda kernels
        self._cmodule = LazyKernelCModule(self.code)

        for name in self._function_names:
            setattr(self, name, KernelFunction(self._cmodule, name))


def initialize_quantization_state():
    """
    Initialize functions related to quantization using some kernel manager
    containing quantization kernel functions.
    """
    # Make manager to init kernels
    kernel_manager = _QuantizationKernelManager("cuda/quantization")

    # Declare quantization/dequantization functions
    def compress_int8_weight_to_int4(
        weight: torch.Tensor,
        kernel_manager: _QuantizationKernelManager = None,
    ):
        with torch.cuda.device(weight.device):
            n, m = weight.size(0), weight.size(1)
            assert m % 2 == 0
            m = m // 2
            out = torch.empty(
                (n, m),
                dtype=torch.int8,
                device=torch.cuda.current_device(),
            )
            stream = torch.cuda.current_stream()

            gridDim = (n, 1, 1)
            blockDim = (min(round_up(m, 32), 1024), 1, 1)

            kernel_manager.int4WeightCompression(
                gridDim,
                blockDim,
                0,
                stream,
                [
                    ctypes.c_void_p(weight.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_int32(n),
                    ctypes.c_int32(m),
                ],
            )
            return out

    def quantize_fp16_weight_to_int8(
        weight: torch.Tensor,
        bit_width: int,
        kernel_manager: _QuantizationKernelManager = None,
    ):
        assert weight.dtype == torch.float16, (
            f"Unsupported dtype to quantize: " f"{weight.dtype}"
        )
        assert bit_width in [4, 8], (
            f"Unsupported bit width for quantization: " f"{bit_width}"
        )

        # NOTE: Quantization only works on GPU (torch.round specifically)
        weight = weight.to(torch.cuda.current_device())

        # absmax quantization
        quantized_weight_scale = (
            weight.abs().max(dim=-1).values / ((2 ** (bit_width - 1)) - 1)
        ).half()
        quantized_weight = torch.round(weight / quantized_weight_scale[:, None]).to(
            torch.int8
        )

        # Further quantize using int4 quantization kernel
        if bit_width == 4:
            quantized_weight = compress_int8_weight_to_int4(
                quantized_weight, kernel_manager
            )

        # Place weight back onto cpu
        quantized_weight = quantized_weight.to("cpu")

        return quantized_weight, quantized_weight_scale

    def extract_any_weight_to_fp16(
        weight: torch.Tensor,
        scale_list: torch.Tensor,
        bit_width: int,
        kernel_manager: _QuantizationKernelManager = None,
    ):
        if bit_width == 8:
            func = kernel_manager.int8WeightExtractionHalf

        elif bit_width == 4:
            func = kernel_manager.int4WeightExtractionHalf

        else:
            raise Exception(f"Unsupported bit-width: {bit_width}")

        with torch.cuda.device(weight.device):
            n, m = weight.size(0), weight.size(1)
            # TODO (mchoi): Device should be `torch.cuda.current_device()`
            out = torch.empty(
                (n, m * (8 // bit_width)),
                dtype=torch.half,
                device=torch.cuda.current_device(),
            )
            stream = torch.cuda.current_stream()

            gridDim = (n, 1, 1)
            blockDim = (min(round_up(m, 32), 1024), 1, 1)

            func(
                gridDim,
                blockDim,
                0,
                stream,
                [
                    ctypes.c_void_p(weight.data_ptr()),
                    ctypes.c_void_p(scale_list.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_int32(n),
                    ctypes.c_int32(m),
                ],
            )
            return out

    quantization_fn = partial(
        quantize_fp16_weight_to_int8, kernel_manager=kernel_manager)
    compression_fn = partial(
        compress_int8_weight_to_int4, kernel_manager=kernel_manager)
    extraction_fn = partial(
        extract_any_weight_to_fp16, kernel_manager=kernel_manager)

    # Expose quantization functions as global functions
    _QUANTIZATION_FUNCTIONS["quantization_fn"] = quantization_fn
    _QUANTIZATION_FUNCTIONS["compression_fn"] = compression_fn
    _QUANTIZATION_FUNCTIONS["extraction_fn"] = extraction_fn


class QuantizedLinear(torch.autograd.Function):
    """
    Quantized linear layer autograd function. Weights are saved in either int4
    or int8, and dynamically extracted back into native fp16 and matmul'd. Note
    that the cached activations for the backwards pass are the quantized
    weights.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        quantized_weight: torch.Tensor,
        quantized_weight_scale: torch.Tensor,
        bit_width: int,
    ):
        ctx.inp_shape = inp.size()
        ctx.weight_shape = quantized_weight.size()
        ctx.bit_width = bit_width

        out_features = quantized_weight.size(0)

        inp = inp.contiguous().view(-1, inp.size(-1))

        try:
             extraction_fn = _QUANTIZATION_FUNCTIONS["extraction_fn"]
        except KeyError:
            raise KeyError(
                "Extraction function does not exist yet! Did "
                "you call `initialize_quantization_state()`?"
            )
        assert quantized_weight.dtype == torch.int8, (f"Wrong dtype detected "
                                                      f"in quantized weight: "
                                                      f"{quantized_weight.dtype}")

        weight = extraction_fn(
            quantized_weight,
            quantized_weight_scale,
            bit_width,
        )
        output = inp.mm(weight.t())

        ctx.save_for_backward(inp, quantized_weight, quantized_weight_scale)

        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # NOTE: Untested and unused for now
        inp, quantized_weight, quantized_weight_scale = ctx.saved_tensors

        try:
            extraction_fn = _QUANTIZATION_FUNCTIONS["extraction_fn"]
        except KeyError:
            raise KeyError(
                "Extraction function does not exist yet! Did "
                "you call `initialize_quantization_state()`?"
            )

        weight = extraction_fn(
            quantized_weight,
            quantized_weight_scale,
            ctx.bit_width,
        )

        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)

        return (
            grad_input.view(ctx.inp_shape),
            grad_weight.view(ctx.weight_shape),
            None,
        )


class QuantizedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        bit_width: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        n, m = self.weight.shape

        if bit_width == 4:
            assert m % 2 == 0
            m //= 2

        self.weight = nn.Parameter(
            torch.empty(
                (n, m),
                device=torch.cuda.current_device(),
                dtype=torch.int8,
            ),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                n,
                device=torch.cuda.current_device(),
                dtype=torch.float16,
            ),
            requires_grad=False)
        self.bit_width = bit_width

    def forward(self, input_: torch.Tensor):
        input_parallel = copy_to_tensor_model_parallel_region(input_)

        output_parallel = QuantizedLinear.apply(
            input_parallel,
            self.weight,
            self.weight_scale,
            self.bit_width,
        )
        if self.skip_bias_add:
            output_bias = self.bias
        else:
            output_parallel = output_parallel + self.bias
            output_bias = None

        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        return output, output_bias


class QuantizedRowParallelLinear(RowParallelLinear):
    def __init__(
        self,
        bit_width: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        n, m = self.weight.shape

        if bit_width == 4:
            assert m % 2 == 0
            m //= 2

        self.weight = nn.Parameter(
            torch.empty(
                (n, m),
                device=torch.cuda.current_device(),
                dtype=torch.int8,
            ),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                n,
                device=torch.cuda.current_device(),
                dtype=torch.float16,
            ),
            requires_grad=False)
        self.bit_width = bit_width

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = QuantizedLinear.apply(
            input_parallel, self.weight, self.weight_scale, self.bit_width
        )
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if self.skip_bias_add:
            output = output_
            output_bias = self.bias
        else:
            output = output_ + self.bias if self.bias is not None else output
            output_bias = None

        return output, output_bias


def quantize_state_dict(
    model_state,
    quantize_module_names,
    bit_width,
):
    """
    Quantize the model state on single GPU/CPU. This should be called in
    the `load_model_ensemble_and_state` function in `checkpoint_utils.py`.
    """
    quantized_weight_scales = {}

    for module_name in quantize_module_names:
        # Get everything before the weight or bias part
        param_type = module_name.split(".")[-1]
        layer_type = ".".join(module_name.split(".")[:-1])

        if param_type != "weight":
            raise NotImplementedError(
                "Recursive submodule quantization not " "supported yet."
            )

        # Cast to fp16 since quantization only supports fp16
        state = model_state[module_name].half()

        # Get full reference to param weight and quantize
        try:
            quantization_fn = _QUANTIZATION_FUNCTIONS["quantization_fn"]
        except KeyError:
            raise KeyError(
                "Quantization function does not exist yet! Did "
                "you call `initialize_quantization_state()`?"
            )
        quantized_state, weight_scale = quantization_fn(state, bit_width)

        # Mutate pass-by-sharing model state
        model_state[module_name] = quantized_state
        quantized_weight_scales[f"{layer_type}.weight_scale"] = weight_scale

    return quantized_weight_scales


def sanity_check_quantized_model(model):
    # TODO (mchoi): Add more extensive tests as necessary
    for module_name, module in model.named_modules():
        param_type = module_name.split(".")[-1]
        #layer_type = ".".join(module_name.split(".")[:-1])

        if param_type.endswith("fc1") or param_type.endswith("fc2") or param_type.endswith("qkv_proj") or param_type.endswith("out_proj"):
            assert module.weight.dtype == torch.int8, (f"Quantization sanity "
                                                       f"check failed, param "
                                                       f"{module_name} had dtype "
                                                       f"{module.weight.dtype}")
            assert module.weight.device != "cpu"
