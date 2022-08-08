from setuptools import setup, find_packages

NAME = 'neuroface'
VERSION = '0.1.0'
DESCRIPTION = 'Human face detection, recognition on video.'
URL = 'https://github.com/mdvdv/neuroface'

setup(
    name=NAME,
    version=VERSION,
    url=URL
    description=DESCRIPTION,
    packages={
        'neuroface',
        'neuroface.emotions',
        'neuroface.face',
        'neuroface.face.comparison',
        'neuroface.face.detection',
        'neuroface.landmarks',
    ],
    install_requires=[
      'torch>=1.12.0',
      'torchvision>=0.13.0',
      'mediapipe>=0.8.10.1',
      'opencv_python>=4.6.0.66',
      'Pillow>=9.2.0',
      'av>=9.2.0',
      'gdown>=4.5.1'
    ],
)
