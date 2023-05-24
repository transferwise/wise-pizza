from setuptools import find_packages, setup


def _read_requirements_file(path: str):
    with open(path) as f:
        return list(map(
            lambda req: req.strip(),
            f.readlines(),
        ))


with open('README.md') as f:
    long_description = f.read()

setup(
    name="wise-pizza",
    version="0.1.1",
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
    install_requires=_read_requirements_file('requirements.txt'),
    extras_require={
        "test": _read_requirements_file('requirements-dev.txt'),
    },
    packages=find_packages(
        include=[
            'wise_pizza',
            'wise_pizza.*'
        ],
        exclude=['tests*'],
    ),
    include_package_data=True,
    keywords='wise-pizza',
)
