import os

# See if Cython is installed
try:
    from Cython.Build import cythonize
# Do nothing if Cython is not available
except ImportError:
    # Got to provide this function. Otherwise, poetry will fail
    def build(setup_kwargs):
        pass
# Cython is installed. Compile
else:
    from setuptools import Extension
    from setuptools.dist import Distribution
    from distutils.command.build_ext import build_ext

    # This function will be executed in setup.py:
    def build(setup_kwargs):
        # The file you want to compile
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

        # Build
        setup_kwargs.update({
            'ext_modules': cythonize(extensions),
            'cmdclass': {'build_ext': build_ext}
        })
