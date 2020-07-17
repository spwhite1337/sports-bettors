from setuptools import setup, find_packages

setup(
    name='sports-bettors',
    version='1.0',
    description='Betting Aid for Select Sports',
    author='Scott P. White',
    author_email='spwhite1337@gmail.com',
    packages=find_packages(),
    entry_points={'console_scripts': [
        'sb_download = sports_bettors.download:download_cli',
        'sb_curate = sports_bettors.curate:curate_data',
        'sb_run_experiments = sports_bettors.experiments:run_experiments',
        'sb_predict = sports_bettors.api:api_cli',
        'sb_generate_predictors = sports_bettors.api:create_predictor_sets',
        'sb_upload = sports_bettors.upload:upload'
    ]},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'requests',
        'ipykernel',
        'scikit-learn',
        'tqdm',
        'pystan',
        'beautifulsoup4',
        'awscli'
    ]
)
