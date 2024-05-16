from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='scsdpy',
      version='0.1.0',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.16.2',
          'pandas>=0.23.4',
          'scipy>=1.4.1',
          'plotly>=5.1.2',
          'flask>=1.1.2',
          'scikit-learn>=0.24.2',
          'networkx>=2.2',
          'matplotlib>=3.2.2',
          'seaborn>=0.11.1',
          'werkzeug>=3.0.3'
      ],
      author='Dr Christopher Kingsbury',
      author_email='ckingsbury@ccdc.cam.ac.uk',
      license='ANTI-1.4',
      url='https://github.com/cjkingsbury/scsd',
      description='SCSD is Python software for the analysis of molecular conformation and deformation in crystal structures',
      platforms=['linux', 'windows', 'osx', 'win32'],
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent"],
      package_data={'scsd': ['data/*', 'data/scsd/*', 'templates/scsd/*', 'static/*', 'scsd_models.json']}
      )
