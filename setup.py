import distutils
import os
import sys
import tempfile
import warnings
from distutils import ccompiler
from distutils.command.build_ext import build_ext
from distutils.errors import CompileError, LinkError
from distutils.sysconfig import customize_compiler
from os.path import join

import setuptools
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


class get_numpy_include:
    """Helper class to determine the numpy include path

    The purpose of this class is to postpone importing numpy until it is
    actually installed, so that the ``get_include()`` method can be invoked.

    """
    def __str__(self):
        import numpy
        return numpy.get_include()


def get_include_dirs():
    """Get all the include directories which may contain headers that we need to
    compile the cython extensions."""
    return (
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Library", "include"),
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

    COMPILER_FLAGS = {
        "unix": {
            "openmp": "-Xpreprocessor -fopenmp" if sys.platform == "darwin" else "-fopenmp",
            "optimize": "-O3",
            "fftw": "-lfftw3",
            "math": "-lm",
            "fast-math": "-ffast-math",
            "native": "-march=native",
        },
        "msvc": {
            "openmp": "/openmp",
            "optimize": "/Ox",
            "fftw": "/lfftw3",
            "math": "",
            "fast-math": "/fp:fast",
            "native": "",
        },
    }

    def build_extensions(self):
        # Automatically append the file extension based on language.
        # ``cythonize`` does this for us automatically, so it's not necessary if
        # that was run
        for extension in extensions:
            for idx, source in enumerate(extension.sources):
                base, ext = os.path.splitext(source)
                if ext == ".pyx":
                    base += ".cpp" if extension.language == "c++" else ".c"
                    extension.sources[idx] = base

        # Optimization compiler/linker flags are added appropriately
        flags = self.COMPILER_FLAGS[self.compiler.compiler_type]
        compile_flags = [flags["math"], flags["optimize"]]
        link_flags = [flags["math"], flags["optimize"]]

        # We don't want the compiler to optimize for system architecture if
        # we're building packages to be distributed by conda-forge, but if the
        # package is being built locally, this is desired
        if not ("AZURE_BUILD" in os.environ or "CONDA_BUILD" in os.environ):
            compile_flags.append(flags["native"])
            link_flags.append(flags["native"])

        # Map any existing compile/link flags into compiler specific ones
        def map_flags(ls):
            return list(map(lambda flag: flags.get(flag, flag), ls))

        for extension in self.extensions:
            extension.extra_compile_args = map_flags(extension.extra_compile_args)
            extension.extra_link_args = map_flags(extension.extra_link_args)

        # We will disable openmp flags if the compiler doesn"t support it. This
        # is only really an issue with OSX clang
        if has_c_library("omp"):
            print("Found openmp. Compiling with openmp flags...")
            compile_flags.append(flags["openmp"]), link_flags.append(flags["openmp"])
        else:
            warnings.warn(
                "You appear to be using a compiler which does not support "
                "openMP, meaning that the library will not be able to run on "
                "multiple cores. Please install/enable openMP to use multiple "
                "cores."
            )

        for extension in self.extensions:
            extension.extra_compile_args.extend(compile_flags)
            extension.extra_link_args.extend(link_flags)

        # Add numpy and system include directories
        for extension in self.extensions:
            extension.include_dirs.extend(get_include_dirs())
            extension.include_dirs.append(get_numpy_include())

        super().build_extensions()


extensions = [
    Extension("openTSNE.quad_tree", ["openTSNE/quad_tree.pyx"]),
    Extension("openTSNE._tsne", ["openTSNE/_tsne.pyx"]),
    Extension("openTSNE.kl_divergence", ["openTSNE/kl_divergence.pyx"]),
]


# Check if we have access to FFTW3 and if so, use that implementation
if has_c_library("fftw3"):
    print("FFTW3 header files found. Using FFTW implementation of FFT.")
    extension_ = Extension(
        "openTSNE._matrix_mul.matrix_mul",
        ["openTSNE/_matrix_mul/matrix_mul_fftw3.pyx"],
        extra_compile_args=["fftw"],
        extra_link_args=["fftw"],
    )
    extensions.append(extension_)
else:
    print("FFTW3 header files not found. Using numpy implementation of FFT.")
    extension_ = Extension(
        "openTSNE._matrix_mul.matrix_mul",
        ["openTSNE/_matrix_mul/matrix_mul_numpy.pyx"],
    )
    extensions.append(extension_)

try:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)
except ImportError:
    pass


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
        "Development Status :: 4 - Beta",
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
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.14.6",
        "scikit-learn>=0.20",
        "scipy",
        "annoy>=1.16.3",
    ],

    ext_modules=extensions,
    cmdclass={"build_ext": CythonBuildExt, "convert_notebooks": ConvertNotebooksToDocs},
)
