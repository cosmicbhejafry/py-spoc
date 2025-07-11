from setuptools import setup

setup()

# commenting out old code for now, the install requires have changed:
# from setuptools import setup, find_packages

# # http://www.diveintopython3.net/packaging.html
# # https://pypi.python.org/pypi?:action=list_classifiers

# with open('README.md', 'r', encoding='utf-8') as file:
#     long_description = file.read()


# install_requires = [
#         'h5py',
#         'scikit-learn',
#         'scipy',
#         'numpy<2.0.0',
#         'pandas',
#         'statsmodels',
#         'pyyaml',
#         'tqdm',
# #        'nitime',
#         'hyppo',
#         'pyEDM==1.15.2.0',
#         'jpype1',
# #        'sktime',
#         'dill',
#         'spectral-connectivity',
#         'torch',
#         'cdt==0.5.23',
#         'oct2py',
# #        'tslearn',
#         'mne==0.23.0',
#         'seaborn',
#         'future'
# ]

# testing_extras = [
#     'pytest==5.4.2',  # unittest.TestCase funkyness, see commit 77c1505ab
# ]


# setup(
#     name='pyspoc',
#     packages=find_packages(),
#     package_data={'': ['run_config/testing.yaml',
#                        'run_config/fabfour_config.yaml',
#                        'lib/jidt/infodynamics.jar',
#                        'lib/PhiToolbox/Phi/phi_comp.m',
#                        'lib/PhiToolbox/Phi/phi_star_Gauss.m',
#                        'lib/PhiToolbox/Phi/phi_G_Gauss.m',
#                        'lib/PhiToolbox/Phi/phi_G_Gauss_AL.m',
#                        'lib/PhiToolbox/Phi/phi_G_Gauss_LBFGS.m',
#                        'lib/PhiToolbox/Phi/phi_comp_probs.m',
#                        'lib/PhiToolbox/Phi/phi_Gauss.m',
#                        'lib/PhiToolbox/utility/I_s_I_s_d.m',
#                        'lib/PhiToolbox/utility/data_to_probs.m',
#                        'lib/PhiToolbox/utility/Gauss/Cov_comp.m',
#                        'lib/PhiToolbox/utility/Gauss/Cov_cond.m',
#                        'lib/PhiToolbox/utility/Gauss/H_gauss.m',
#                        'lib/PhiToolbox/utility/Gauss/logdet.m',
#                        'data/cml.npy',
#                        'data/forex.npy',
#                        'data/standard_normal.npy',
#                        'data/cml7.npy']},
#     include_package_data=True,
#     version='1.0.1',
#     description='Library for statistical analysis of static datasets.',
#     author='Garry S. Cotton',
#     author_email='garry.s.cotton@gmail.com',
#     url='https://github.com/garry-cotton/pyss',
#     long_description=long_description,
#     classifiers=[
#         "Programming Language :: Python",
#         "Programming Language :: Python :: 3",
#         "Development Status :: 1 - Planning",
#         "Operating System :: POSIX :: Linux",
#         "Intended Audience :: Science/Research",
#         "Environment :: Console",
#         "Environment :: Other Environment",
#         "Topic :: Scientific/Engineering :: Physics",
#         "Topic :: Scientific/Engineering :: Bio-Informatics",
#         "Topic :: Scientific/Engineering :: Information Analysis",
#         "Topic :: Scientific/Engineering :: Medical Science Apps.",
#     ],
#     install_requires=install_requires,
#     extras_require={'testing': testing_extras}
# )
