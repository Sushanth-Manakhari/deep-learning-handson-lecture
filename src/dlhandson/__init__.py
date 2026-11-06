"""dlhandson — the code behind the "Deep Learning, Hands-On" guest lecture.

The notebooks show every idea step by step; this package holds the same code in
a reusable, tested form. That split mirrors industry practice:

    notebook  -> explore, explain, visualise   (great for learning, bad for production)
    package   -> import, test, version, deploy (what teams actually ship)

Modules
-------
device       pick CUDA / Apple MPS / CPU, seeds, environment detection
config       one dataclass per task with sensible defaults + hardware "profiles"
data_vision  Beans leaf-disease images (download with fallbacks, transforms, loaders)
data_text    Banking77 customer-intent text (loading, tokenisation, dynamic padding)
models       pretrained backbones, freezing, new heads, parameter counting
train        the canonical training loop (+ mixed precision, clipping, NaN guard)
debug_tools  overfit-one-batch, OOM-safe batch search, vanishing/exploding demos
ood          stress tests, softmax confidence and "route to a human" thresholds
viz          plots used in the notebooks and slides
tracking     TensorBoard (+ optional Weights & Biases) experiment logging
demo_app     a Gradio web app so anyone can try the models from a phone
timing       section timer used to calibrate the lecture schedule
"""

__version__ = "1.0.0"

CLASS_NAMES_BEANS = ["angular_leaf_spot", "bean_rust", "healthy"]
