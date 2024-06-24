from setuptools import setup, find_packages

setup(
  name = 'fluoriclogppka',
  packages = find_packages(), 
  include_package_data = True,

  version = '0.2.7',
  license = 'MIT',
  description = 'Tool for pKa, logP prediction',
  author = 'Blackthorn.ai',
  url = 'https://github.com/blackthorn-ai/fluoricLogPpKa',
  download_url = 'https://github.com/blackthorn-ai/fluoricLogPpKa/dist/fluoriclogppka-0.2.7.tar.gz',
  keywords = ['pKa', 'logP', 'tool'],
  
  install_requires=[ 
          'certifi==2024.2.2',
          'charset-normalizer==3.3.2',
          'future==1.0.0',
          'h2o==3.44.0.3',
          'idna==3.6',
          'mordred==1.2.0',
          'networkx==2.8.8',
          'numpy==1.26.4',
          'openpyxl==3.1.4',
          'packaging==24.0',
          'pandas>=2.2.0',
          'pillow>=9.2.0',
          'python-dateutil==2.8.2',
          'pytz==2024.1',
          'rdkit==2022.9.5',
          'requests==2.31.0',
          'six==1.16.0',
          'tabulate==0.9.0',
          'tzdata==2024.1',
          'urllib3==2.2.1',
          'torch==2.0.0',
          'torchvision==0.15.1',
          'dgl==1.1.2',
          'dgllife==0.3.2'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.11',
  ],
)
