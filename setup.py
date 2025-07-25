from setuptools import setup, find_packages

setup(
    name='srm_detection_pipeline',
    version='0.1.0',
    description='A package for SRM detection and analysis',
    author='Ihssene Brahimi, Moustafa Amine Bezzahi',
    author_email='ji_brahimi@esi.dz',
    url='https://github.com/yourusername/srm_detection_pipeline',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'monai',
        'tqdm',
        'matplotlib',
        'scipy',
        'imageio',
        'nrrd',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'run-srm=srm_detection.main:main',
        ],
    },
)
