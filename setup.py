from setuptools import setup, find_packages

setup(name='scsd',
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
      license='ANTI-1.4',
      description='Scientific Data Analysis Toolkit',
      long_description='SCSD is Python software for the analysis of molecular conformation and deformation in crystal structures',
      platforms=['linux', 'windows', 'osx', 'win32'],
      package_data={'scsd': ['data/*', 'data/scsd/*', 'templates/scsd/*', 'static/*', 'scsd_models.json']}
      )
