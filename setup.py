from setuptools import find_packages, setup

setup(
    name="weavenn",
    version="0.0.1",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="Density-aware k-nearest neighbor graph from cloud points",
    url="https://github.com/kerighan/weavenn",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["networkx", "numpy", "scikit-learn", "python-louvain"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
