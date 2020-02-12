import io
import os

from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "_youtokentome_cython",
        [
            "youtokentome/cpp/yttm.pyx",
            "youtokentome/cpp/bpe.cpp",
            "youtokentome/cpp/utils.cpp",
            "youtokentome/cpp/utf8.cpp",
        ],
        extra_compile_args=["-std=c++11", "-pthread", "-O3"],
        language="c++",
    )
]

with io.open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
        encoding="utf-8",
) as f:
    LONG_DESCRIPTION = "\n" + f.read()

setup(
    name="youtokentome",
    version="1.0.6",
    packages=find_packages(),
    description="Unsupervised text tokenizer focused on computational efficiency",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/vkcom/youtokentome",
    python_requires=">=3.5.0",
    install_requires=["Click>=7.0"],
    entry_points={"console_scripts": ["yttm = youtokentome.yttm_cli:main"]},
    author="Ivan Belonogov",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Cython",
        "Programming Language :: C++",
    ],
    ext_modules=cythonize(extensions),
)

