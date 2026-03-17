from setuptools import setup, find_packages

setup(
    name="xray-enhancement",
    version="0.1.0",
    description="Low-Data Medical Image Enhancement for X-ray Images",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.21.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
        "albumentations>=1.3.0",
        "scipy>=1.11.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
