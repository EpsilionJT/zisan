from setuptools import setup, find_packages

setup(name='zisan',
      version='1.0.19',
      description='Integrated CV basic application development kit based on Pytorch',
      author='JintuZheng',
      author_email='jintuzheng@outlook.com',
      url='http://www.jintupersonal.com/zisan/',
      license='MIT',
      keywords='zisan instanceSegmentation Object detecttion cv',
      packages=find_packages(),
      install_requires=['numpy', 'torch', 'pillow','matplotlib','torchvision','opencv-python','scikit-image','scipy'],
      python_requires='>=3.5'
     )