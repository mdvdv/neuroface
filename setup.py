from setuptools import setup, find_packages

setup(
    name='neuroface',
    namespace_packages=['neuroface'],
    version='0.1.0',
    url='https://github.com/mdvdv/neuroface',
    description='Human face detection, recognition on video.',
    packages=find_packages(),    
    install_requires=[
      'torch>=1.12.0',
      'torchvision>=0.13.0',
      'mediapipe>=0.8.10.1',
      'opencv_python>=4.6.0.66',
      'numpy>=1.23.1',
      'Pillow>=9.2.0',
      'av>=9.2.0',
      'gdown>=4.5.1],
)
