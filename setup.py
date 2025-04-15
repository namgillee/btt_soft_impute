from setuptools import setup, find_packages

setup(
    name='data_analysis_tool',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy'
    ],
    author='Namgil Lee',
    author_email='namgil.lee' '@' 'kangwon.ac.kr',
    description='test tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/namgillee/BTT-SCD',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
