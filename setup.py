from setuptools import setup, find_packages

setup(
    name='iowa-football',
    version='1.0',
    description='Betting Aid for College Football',
    author='Scott P. White',
    author_email='spwhite1337@gmail.com',
    packages=find_packages(),
    entry_points={'console_scripts': [
    ]},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'requests',
        'plotly',
        'dash',
        'ipykernel',
        'scikit-learn',
        'statsmodels',
        'tqdm'
    ]
)
