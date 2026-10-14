from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sisa-unlearning",
    version="1.0.0",
    author="Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Shubham Singh, Swarup Ranjan Behera, Vandana Rajan, Muskaan Singh, Arun Balaji Buduru, Rajesh Sharma",
    description="SISA++: Machine Unlearning for Paralinguistic Speech Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helix-IIIT-Delhi/SISA-Unlearning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
