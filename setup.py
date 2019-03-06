import sem_policy_opt
from setuptools import setup
from setuptools import find_packages

setup(name='sem_policy_opt',
      version=sem_policy_opt.__version__,
      description='Code used to show how domain knowledge can be combined with RL to solve common business problems',
      url='http://github.com/dansbecker/sem_policy_opt',
      author='Dan Becker',
      author_email='danbecker@google.com',
      packages=find_packages(),
      zip_safe=True)
