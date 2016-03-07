from setuptools import setup, find_packages

setup(
    name='PyWANN',
    version='0.1',
    description='Python Weightless Artificial Neural Networks',
    author='Fabricio Firmino, Fabio Rangel',
    author_email='firminodefaria@gmail.com, fabio.mrangel@gmail.com',
    url='http://ppgi.ufrj.br/',
    packages=find_packages(),
    long_description="Python Weightless Artificial Neural Networks lib",
    classifiers=["Weightless neural networks", "machine learning"],
    license='Apache License',
    install_requires=['setuptools', 'numpy', 'pyyaml'],

    package_dir={'WiSARD':    './PyWANN/WiSARD', \
                 'BloomWiSARD': './PyWANN/BloomWiSARD'}

    )



s