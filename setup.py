from setuptools import setup, find_packages

setup(
    name="gdt",
    version="0.1.0",
    description=(
        "Graph-Differential Transformer: A Unified Multimodal Architecture "
        "for Vision-Language Tasks"
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.35.0",
        "timm>=0.9.0",
        "pycocotools>=2.0.6",
        "pycocoevalcap>=1.2",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "gdt-train=gdt.scripts.train:main",
            "gdt-evaluate=gdt.scripts.evaluate:main",
        ],
    },
)
