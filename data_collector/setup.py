from setuptools import find_packages, setup

package_name = 'data_collector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pandas', 'numpy'],
    zip_safe=True,
    maintainer='nisara',
    maintainer_email='sarawgi.nikita@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kuka_data_recorder = data_collector.kuka_data_recorder:main',
            'joint_cartesian_converter = data_collector.joint_cartesian_converter:main'
        ],
    },
)
