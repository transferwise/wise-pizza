from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name="wise-pizza",
    version="0.2.2",
    description="A library to find and visualise the most interesting slices in multidimensional data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Wise",
    url='https://github.com/transferwise/wise-pizza',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        "ipython",
        "kaleido",
        "numpy",
        "pandas",
        "pytest",
        "plotly",
        "scikit_learn",
        "scipy>=1.8.0",
        "tqdm",
        "cloudpickle",
        "pivottablejs",
        "streamlit==1.28.0"
    ],
    extras_require={
        "test": [
            "flake8",
            "pytest",
            "pytest-cov"
        ],
    },
    packages=find_packages(
        include=[
            'wise_pizza',
            'wise_pizza.*'
        ],
        exclude=['tests*'],
    ),
    entry_points={
        'console_scripts': [
            'run_wise_pizza_streamlit = wise_pizza.run_streamlit_app_entry_point:main',
        ],
    },
    include_package_data=True,
    keywords='wise-pizza',
)
