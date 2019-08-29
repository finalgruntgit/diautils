from setuptools import setup

setup(name='diautils',
      version='1.3.6',
      description='ML Utils',
      author='Eric Renault',
      author_email='eric.renaul.info@gmail.com',
      license='MIT',
      packages=['diautils'],
      zip_safe=False,
      install_requires=[
            'numpy==1.15.4',
            'numba==0.41.0'
      ]
)
