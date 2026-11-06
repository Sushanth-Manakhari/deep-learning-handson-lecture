"""Every external dataset, model and paper used in this project — one place.

The README, technical design document, notebooks and slides all cite these.
`sources_table()` returns them as a DataFrame (shown in the notebook appendix).
"""
from __future__ import annotations

DATASETS = [
    {"name": "Beans (leaf disease photos)", "used_for": "Ticket LEAF-101 (vision)",
     "url": "https://huggingface.co/datasets/AI-Lab-Makerere/beans",
     "original": "https://github.com/AI-Lab-Makerere/ibean/",
     "licence": "MIT", "size": "1,295 images (1,034/133/128), 3 classes, ~180 MB parquet; ~25 MB resized pack",
     "how": "GitHub Release asset beans_224.zip (SHA-256 checked) → fallback: Hugging Face Hub parquet files"},
    {"name": "Banking77 (customer intents)", "used_for": "Ticket SUPPORT-202 (NLP)",
     "url": "https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data",
     "original": "https://arxiv.org/abs/2003.04807",
     "licence": "CC BY 4.0", "size": "13,083 messages (10,003/3,080), 77 intents, ~1 MB",
     "how": "committed in data/banking77/ (re-downloaded from GitHub if missing)"},
    {"name": "scikit-image sample photos (coffee cup, cat)", "used_for": "Out-of-distribution stress test",
     "url": "https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data",
     "original": "coffee.png (Rachel Michetti), chelsea.png (Stefan van der Walt)",
     "licence": "CC0", "size": "2 images", "how": "committed in data/stress_test/non_leaf/"},
    {"name": "MedMNIST v2 — PneumoniaMNIST (take-home)", "used_for": "Extension B1 (health)",
     "url": "https://medmnist.com/", "original": "https://doi.org/10.1038/s41597-022-01721-8",
     "licence": "CC BY 4.0", "size": "5,856 chest X-rays", "how": "pip install medmnist (not used in class)"},
    {"name": "SMS Spam Collection (take-home)", "used_for": "Extension B2 (security)",
     "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
     "original": "https://huggingface.co/datasets/ucirvine/sms_spam",
     "licence": "CC BY 4.0", "size": "5,574 messages", "how": "Hugging Face datasets (not used in class)"},
]

MODELS = [
    {"name": "ResNet-50, ImageNet-1K V2 weights (torchvision)", "params": "25.6M", "download": "~98 MB",
     "url": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html",
     "weights": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth", "licence": "BSD-3-Clause (torchvision)",
     "paper": "https://arxiv.org/abs/1512.03385", "used_for": "Vision backbone (GPU / Apple MPS)"},
    {"name": "ResNet-18, ImageNet-1K V1 weights (torchvision)", "params": "11.7M", "download": "~45 MB",
     "url": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html",
     "weights": "https://download.pytorch.org/models/resnet18-f37072fd.pth", "licence": "BSD-3-Clause (torchvision)",
     "paper": "https://arxiv.org/abs/1512.03385", "used_for": "Vision backbone in CPU FAST mode"},
    {"name": "ViT-B/16, ImageNet-1K V1 weights (torchvision)", "params": "86.6M", "download": "~330 MB",
     "url": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html",
     "weights": "https://download.pytorch.org/models/vit_b_16-c867db91.pth", "licence": "BSD-3-Clause (torchvision)",
     "paper": "https://arxiv.org/abs/2010.11929", "used_for": "Instructor demo: Vision Transformer swap"},
    {"name": "DistilBERT base uncased (Hugging Face)", "params": "66M", "download": "~260 MB",
     "url": "https://huggingface.co/distilbert/distilbert-base-uncased", "weights": "model.safetensors on the Hub",
     "licence": "Apache-2.0", "paper": "https://arxiv.org/abs/1910.01108", "used_for": "Text backbone"},
]

PAPERS = [
    ("Deep Residual Learning (ResNet), He et al. 2015", "https://arxiv.org/abs/1512.03385"),
    ("Delving Deep into Rectifiers (He initialisation), He et al. 2015", "https://arxiv.org/abs/1502.01852"),
    ("Attention Is All You Need (Transformer), Vaswani et al. 2017", "https://arxiv.org/abs/1706.03762"),
    ("BERT, Devlin et al. 2018", "https://arxiv.org/abs/1810.04805"),
    ("DistilBERT, Sanh et al. 2019", "https://arxiv.org/abs/1910.01108"),
    ("An Image is Worth 16x16 Words (ViT), Dosovitskiy et al. 2020", "https://arxiv.org/abs/2010.11929"),
    ("Decoupled Weight Decay (AdamW), Loshchilov & Hutter 2017", "https://arxiv.org/abs/1711.05101"),
    ("Mixed Precision Training, Micikevicius et al. 2017", "https://arxiv.org/abs/1710.03740"),
    ("A Baseline for Detecting Misclassified and OOD Examples (MSP), Hendrycks & Gimpel 2016",
     "https://arxiv.org/abs/1610.02136"),
    ("On Calibration of Modern Neural Networks (temperature scaling), Guo et al. 2017",
     "https://arxiv.org/abs/1706.04599"),
    ("Model Cards for Model Reporting, Mitchell et al. 2018", "https://arxiv.org/abs/1810.03993"),
    ("Efficient Intent Detection with Dual Sentence Encoders (Banking77), Casanueva et al. 2020",
     "https://arxiv.org/abs/2003.04807"),
    ("LoRA: Low-Rank Adaptation of LLMs, Hu et al. 2021", "https://arxiv.org/abs/2106.09685"),
    ("Generative Adversarial Networks, Goodfellow et al. 2014", "https://arxiv.org/abs/1406.2661"),
    ("Denoising Diffusion Probabilistic Models, Ho et al. 2020", "https://arxiv.org/abs/2006.11239"),
]

TOOLS = [
    ("PyTorch", "https://pytorch.org"), ("torchvision", "https://docs.pytorch.org/vision/stable/"),
    ("Hugging Face Transformers", "https://huggingface.co/docs/transformers"),
    ("Hugging Face Hub", "https://huggingface.co"), ("Gradio", "https://www.gradio.app"),
    ("TensorBoard", "https://www.tensorflow.org/tensorboard"), ("Weights & Biases", "https://wandb.ai"),
    ("safetensors", "https://huggingface.co/docs/safetensors"), ("Google Colab", "https://colab.research.google.com"),
    ("Kaggle Notebooks", "https://www.kaggle.com/code"),
]


def sources_table():
    import pandas as pd

    rows = [{"type": "dataset", "name": d["name"], "licence": d["licence"], "link": d["url"]} for d in DATASETS]
    rows += [{"type": "model", "name": m["name"], "licence": m["licence"], "link": m["url"]} for m in MODELS]
    return pd.DataFrame(rows)
