from setuptools import setup, find_packages

setup(
    name='college-football',
    version='1.0',
    description='Betting Aid for College Football',
    author='Scott P. White',
    author_email='spwhite1337@gmail.com',
    packages=find_packages(),
    entry_points={'console_scripts': [
        'cf_download = college_football.download:download_cli',
        'cf_curate = college_football.curate:curate_data',
        'cf_run_experiments = college_football.experiments:run_experiments'
    ]},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'requests',
        'plotly',
        'dash',
        'ipykernel',
        'scikit-learn',
        'statsmodels',
        'tqdm',
        'pystan',
        'beautifulsoup4'
    ]
)
