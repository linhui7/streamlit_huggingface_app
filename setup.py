from setuptools import setup, find_packages

setup(
    name="streamlit_huggingface_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.22.0",
        "transformers>=4.28.0",
        "torch>=2.0.0",
        "accelerate>=0.18.0",
    ],
    python_requires=">=3.8",
)
