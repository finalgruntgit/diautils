from setuptools import setup, find_packages

setup(name='diautils',
      version='1.6.0',
      description='ML Utils',
      author='Eric Renault',
      author_email='eric.renaul.info@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
            'numpy',
            'numba',
            'airspeed',
            'matplotlib==3.1.0',
            'sklearn',
            'sortedcontainers',
            'networkx',
            'pillow'
      ]
)
