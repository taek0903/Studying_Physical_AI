from setuptools import find_packages, setup

package_name = 'py_fibonacci'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bb',
    maintainer_email='rokey7@email.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fibonacci_action_server = py_fibonacci.fibonacci_action_server:main',
            'fibonacci_action_client = py_fibonacci.fibonacci_action_client:main'
        ],
    },
)
