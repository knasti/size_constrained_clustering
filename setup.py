1 from setuptools import find_packages, Extension
2
3 try:
4     from setuptools import setup
5 except ImportError:
6     from distutils.core import setup
7
8 import os
9
10 this_directory = os.path.abspath(os.path.dirname(__file__))
11 with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
12     long_description = f.read()
13
14 # This is no longer needed and was causing warnings. It's handled by pyproject.toml now.
15 # dist.Distribution().fetch_build_eggs(["cython>=0.27", "numpy>=1.13"])
16
17 try:
18     from numpy import get_include
19 except ImportError:
20     def get_include():
21         from numpy import get_include
22         return get_include()
23
24 try:
25     from Cython.Build import cythonize
26 except ImportError:
27     print("! Could not import Cython !")
28     cythonize = None
29
30 def no_cythonize(extensions, **_ignore):
31     for extension in extensions:
32         sources = []
33         for sfile in extension.sources:
34             path, ext = os.path.splitext(sfile)
35             if ext in (".pyx", ".py"):
36                 if extension.language == "c++":
37                     ext = ".cpp"
38                 else:
39                     ext = ".c"
40                 sfile = path + ext
41             sources.append(sfile)
42         extension.sources[:] = sources
43     return extensions
44
45 # FIX: Use relative paths instead of absolute paths
46 extensions = [
47     Extension(
48         "size_constrained_clustering.k_means_constrained.mincostflow_vectorized_",
49         [
50             "size_constrained_clustering/k_means_constrained/mincostflow_vectorized_.pyx",
51         ],
52         include_dirs=[get_include()],
53     ),
54     Extension(
55         "size_constrained_clustering.sklearn_import.cluster._k_means",
56         [
57             "size_constrained_clustering/sklearn_import/cluster/_k_means.pyx"
58         ],
59         include_dirs=[get_include()],
60     ),
61     Extension(
62         "size_constrained_clustering.sklearn_import.metrics.pairwise_fast",
63         [
64             "size_constrained_clustering/sklearn_import/metrics/pairwise_fast.pyx",
65         ],
66         include_dirs=[get_include()],
67     ),
68     Extension(
69         "size_constrained_clustering.sklearn_import.utils.sparsefuncs_fast",
70         [
71             "size_constrained_clustering/sklearn_import/utils/sparsefuncs_fast.pyx",
72         ],
73         include_dirs=[get_include()],
74     ),
75 ]
76
77 CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 1))) and cythonize is not None
78
79 if CYTHONIZE:
80     compiler_directives = {"language_level": 3, "embedsignature": True}
81     extensions = cythonize(extensions, compiler_directives=compiler_directives)
82 else:
83     extensions = no_cythonize(extensions)
84
85 with open("requirements.txt") as fp:
86     install_requires = fp.read().strip().split("\n")
87
88 VERSION = "0.1.3"
89 LICENSE = "MIT"
90 setup(
91     ext_modules=extensions,
92     version=VERSION,
93     # setup_requires is also deprecated and handled by pyproject.toml
94     # setup_requires=["cython", "numpy"],
95     install_requires=install_requires,
96     name="size_constrained_clustering",
97     description="Size Constrained Clustering solver",
98     long_description=long_description,
99     long_description_content_type="text/markdown",
100     url="https://github.com/jingw2/size_constrained_clustering",
101     author="Jing Wang",
102     author_email="jingw2@foxmail.com",
103     license=LICENSE,
104     packages=find_packages(),
105     python_requires=">=3.6",
106 )
