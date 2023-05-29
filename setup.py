from glob import glob
import pybind11
from setuptools import Extension, find_packages, setup


ext_modules = [
    Extension(
        '_weavenn',
        sorted(glob("src/*.cpp")),
        language='c++',
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(True), ],
        extra_compile_args=["-Ofast", "-std=c++17"])
]

setup(
    name="weavenn",
    version="1.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["networkx", "numpy", "scikit-learn", "numba", "hnswlib"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=ext_modules
)
