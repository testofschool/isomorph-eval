from setuptools import setup, find_packages

setup(
    name="isomorph-eval",
    version="1.0.0",
    author="Jung Min Kang",
    author_email="gangjeongmin23@gmail.com",
    description="Separating Reasoning from Recall in LLM Evaluation via Structurally Equivalent Benchmarks and Item Response Theory",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/testofschool/isomorph-eval",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
