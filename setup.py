from setuptools import setup

setup(name='diautils',
      version='1.5.0',
      description='ML Utils',
      author='Eric Renault',
      author_email='eric.renaul.info@gmail.com',
      license='MIT',
      packages=['diautils'],
      zip_safe=False,
      install_requires=[
            'numpy',
            'numba',
            'airspeed',
            'matplotlib==3.1.0',
            'sklearn',
            'sortedcontainers',
            'networkx',
      ]
)
