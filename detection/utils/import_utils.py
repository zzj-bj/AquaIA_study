# Z: This script centralizes DALI imports, determines whether DALI can be used in current env
import torch

try:
    # Z: DALI is NVIDIA's high-performance data loading and image processing library
    # commonly used to decode resize normalize images on the GPU
    # reducing the pressure on CPU DataLoader
    import nvidia.dali as dali
    # Z: ndd non pipeline style but similar functionnality
    import nvidia.dali.experimental.dynamic as ndd  # noqa: F401
    # Z: DALIRaggedIterator: Wrap the output of DALI pipeline into a batch that PyTorch can iterate through
    # Z: LastBatchPolicy: Control how the last batch is handled
    from nvidia.dali.plugin.pytorch import DALIRaggedIterator, LastBatchPolicy  # noqa: F401

    # Import DALI components to avoid having to import them in every file that uses DALI
    # Z: support operation fct (load, resize, transpose...)
    fn = dali.fn
    # Z: for define DALI data processing pipeline
    pipeline_def = dali.pipeline_def
    # Z: some types, enumerations and parameter options for DALI
    types = dali.types
    if not torch.cuda.is_available():
        DALI_AVAILABLE = False
    else:
        DALI_AVAILABLE = True
except ImportError:

    # Z: define a dummy function to avoid import errors when DALI is not available
    # the decorator pipeline_def will return the function decorated itself
    def pipeline_def(func):
        return func

    DALI_AVAILABLE = False
    fn = None
    types = None
    ndd = None
    DALIRaggedIterator = None
    LastBatchPolicy = None

# Temporary benchmark override: force the PyTorch/PIL data-loading path on VM
# DALI_AVAILABLE = False
