from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'py_my_parameter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'lunch'), glob('launch/**'))
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taek',
    maintainer_email='taek@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'py_my_parameter = py_my_parameter.parameter_node:main'
        ],
    },
)
