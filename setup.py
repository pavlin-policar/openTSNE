import distutils
import os
import platform
import sys
import tempfile
import warnings
from distutils import ccompiler
from distutils.errors import CompileError, LinkError
from distutils.sysconfig import customize_compiler
from os.path import join

import setuptools
from Cython.Distutils.build_ext import new_build_ext as build_ext
from setuptools import setup, Extension


class ConvertNotebooksToDocs(distutils.cmd.Command):
    description = "Convert the example notebooks to reStructuredText that will" \
                  "be available in the documentation."

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import nbconvert
        from os.path import join

        exporter = nbconvert.RSTExporter()
        writer = nbconvert.writers.FilesWriter()

        files = [
            join("examples", "01_simple_usage.ipynb"),
            join("examples", "02_advanced_usage.ipynb"),
            join("examples", "03_preserving_global_structure.ipynb"),
            join("examples", "04_large_data_sets.ipynb"),
        ]
        target_dir = join("docs", "source", "examples")

        for fname in files:
            self.announce(f"Converting {fname}...")
            directory, nb_name = fname.split("/")
            nb_name, _ = nb_name.split(".")
            body, resources = exporter.from_file(fname)
            writer.build_directory = join(target_dir, nb_name)
            writer.write(body, resources, nb_name)


def get_numpy_include():
    import numpy
    return numpy.get_include()


def get_include_dirs():
    """Get include dirs for the compiler."""
    return (
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Library", "include"),
    )


def get_library_dirs():
    """Get library dirs for the compiler."""
    return (
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "Library", "lib"),
    )


def has_c_library(library, extension=".c"):
    """Check whether a C/C++ library is available on the system to the compiler.

    Parameters
    ----------
    library: str
        The library we want to check for e.g. if we are interested in FFTW3, we
        want to check for `fftw3.h`, so this parameter will be `fftw3`.
    extension: str
        If we want to check for a C library, the extension is `.c`, for C++
        `.cc`, `.cpp` or `.cxx` are accepted.

    Returns
    -------
    bool
        Whether or not the library is available.

    """
    with tempfile.TemporaryDirectory(dir=".") as directory:
        name = join(directory, "%s%s" % (library, extension))
        with open(name, "w") as f:
            f.write("#include <%s.h>\n" % library)
            f.write("int main() {}\n")

        # Get a compiler instance
        compiler = ccompiler.new_compiler()
        # Configure compiler to do all the platform specific things
        customize_compiler(compiler)
        # Add conda include dirs
        for inc_dir in get_include_dirs():
            compiler.add_include_dir(inc_dir)
        assert isinstance(compiler, ccompiler.CCompiler)

        try:
            # Try to compile the file using the C compiler
            compiler.link_executable(compiler.compile([name]), name)
            return True
        except (CompileError, LinkError):
            return False


class CythonBuildExt(build_ext):
    def build_extensions(self):
        extra_compile_args = []
        extra_link_args = []

        # Optimization compiler/linker flags are added appropriately
        compiler = self.compiler.compiler_type
        if compiler == "unix":
            extra_compile_args += ["-O3"]
        elif compiler == "msvc":
            extra_compile_args += ["/Ox", "/fp:precise"]  # can't use fp:fast because we use inf

        if compiler == "unix":
            # https://stackoverflow.com/questions/22931147/stdisinf-does-not-work-with-ffast-math-how-to-check-for-infinity
            extra_compile_args += [
                "-ffast-math",
                "-fno-finite-math-only",  # we use infinity
                "-fno-associative-math",
            ]

        # Annoy specific flags
        annoy_ext = None
        for extension in extensions:
            if "annoy.annoylib" in extension.name:
                annoy_ext = extension
        assert annoy_ext is not None, "Annoy extension not found!"

        if compiler == "unix":
            annoy_ext.extra_compile_args += ["-std=c++14"]
            annoy_ext.extra_compile_args += ["-DANNOYLIB_MULTITHREADED_BUILD"]
        elif compiler == "msvc":
            annoy_ext.extra_compile_args += ["/std:c++14"]

        # Set minimum deployment version for MacOS
        if compiler == "unix" and platform.system() == "Darwin":
            macos_deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "10.12")
            extra_compile_args += [f"-mmacosx-version-min={macos_deployment_target}"]
            extra_link_args += ["-stdlib=libc++", f"-mmacosx-version-min={macos_deployment_target}"]

        # We don't want the compiler to optimize for system architecture if
        # we're building packages to be distributed by conda-forge, but if the
        # package is being built locally, this is desired
        if not ("AZURE_BUILD" in os.environ or "CONDA_BUILD" in os.environ):
            if platform.machine() == "ppc64le":
                extra_compile_args += ["-mcpu=native"]
            if platform.machine() == "x86_64":
                extra_compile_args += ["-march=native"]

        # We will disable openmp flags if the compiler doesn't support it. This
        # is only really an issue with OSX clang
        if has_c_library("omp"):
            print("Found openmp. Compiling with openmp flags...")
            if platform.system() == "Darwin" and compiler == "unix":
                extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                extra_link_args += ["-lomp"]
            elif compiler == "unix":
                extra_compile_args += ["-fopenmp"]
                extra_link_args += ["-fopenmp"]
            elif compiler == "msvc":
                extra_compile_args += ["/openmp"]
                extra_link_args += ["/openmp"]
        else:
            warnings.warn(
                "You appear to be using a compiler which does not support "
                "openMP, meaning that the library will not be able to run on "
                "multiple cores. Please install/enable openMP to use multiple "
                "cores."
            )

        for extension in self.extensions:
            extension.extra_compile_args += extra_compile_args
            extension.extra_link_args += extra_link_args

        # Add numpy and system include directories
        for extension in self.extensions:
            extension.include_dirs.extend(get_include_dirs())
            extension.include_dirs.append(get_numpy_include())

        # Add numpy and system include directories
        for extension in self.extensions:
            extension.library_dirs.extend(get_library_dirs())

        super().build_extensions()


# Prepare the Annoy extension
# Adapted from annoy setup.py
# Various platform-dependent extras
extra_compile_args = []
extra_link_args = []

annoy_path = "openTSNE/dependencies/annoy/"
annoy = Extension(
    "openTSNE.dependencies.annoy.annoylib",
    [annoy_path + "annoymodule.cc"],
    depends=[annoy_path + f for f in ["annoylib.h", "kissrandom.h", "mman.h"]],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# Other extensions
extensions = [
    Extension("openTSNE.quad_tree", ["openTSNE/quad_tree.pyx"], language="c++"),
    Extension("openTSNE._tsne", ["openTSNE/_tsne.pyx"], language="c++"),
    Extension("openTSNE.kl_divergence", ["openTSNE/kl_divergence.pyx"], language="c++"),
    annoy,
]


# Check if we have access to FFTW3 and if so, use that implementation
if has_c_library("fftw3"):
    print("FFTW3 header files found. Using FFTW implementation of FFT.")
    extension_ = Extension(
        "openTSNE._matrix_mul.matrix_mul",
        ["openTSNE/_matrix_mul/matrix_mul_fftw3.pyx"],
        libraries=["fftw3"],
        language="c++",
    )
    extensions.append(extension_)
else:
    print("FFTW3 header files not found. Using numpy implementation of FFT.")
    extension_ = Extension(
        "openTSNE._matrix_mul.matrix_mul",
        ["openTSNE/_matrix_mul/matrix_mul_numpy.pyx"],
        language="c++",
    )
    extensions.append(extension_)


def readme():
    with open("README.rst", encoding="utf-8") as f:
        return f.read()


# Read in version
__version__: str = ""  # This is overridden by the next line
exec(open(os.path.join("openTSNE", "version.py")).read())

setup(
    name="openTSNE",
    description="Extensible, parallel implementations of t-SNE",
    long_description=readme(),
    version=__version__,
    license="BSD-3-Clause",

    author="Pavlin Poličar",
    author_email="pavlin.g.p@gmail.com",
    url="https://github.com/pavlin-policar/openTSNE",
    project_urls={
        "Documentation": "https://opentsne.readthedocs.io/",
        "Source": "https://github.com/pavlin-policar/openTSNE",
        "Issue Tracker": "https://github.com/pavlin-policar/openTSNE/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "License :: OSI Approved",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    packages=setuptools.find_packages(include=["openTSNE", "openTSNE.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.16.6",
        "scikit-learn>=0.20",
        "scipy",
    ],
    extras_require={
        "hnsw": "hnswlib~=0.4.0",
        "pynndescent": "pynndescent~=0.5.0",
    },
    ext_modules=extensions,
    cmdclass={"build_ext": CythonBuildExt, "convert_notebooks": ConvertNotebooksToDocs},
)
