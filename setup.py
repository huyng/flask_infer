from setuptools import setup

VERSION = '0.1'

setup(
    name='flask_predict',
    description="Flask predict",
    author_email='huyng@oath.com',
    version=VERSION,
    packages=['flask_predict', 'flask_predict.frontend'],
    package_data={'flask_predict.frontend': 'flask_predict/frontend/*'},
    include_package_data=True,
    install_requires=[
        'Flask',
    ],
    zip_safe=False,
)
