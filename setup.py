from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'openarm_data_collection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rômulo Maciel',
    maintainer_email='rmul.maciel@gmail.com',
    description='Automated data collection for OpenArm',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'episode_recorder = openarm_data_collection.episode_recorder:main',
            'bimanual_teleop = openarm_data_collection.bimanual_teleop:main',
            'home_mover = openarm_data_collection.home_mover:main',
            'validate_dataset = openarm_data_collection.validate_dataset:main',
            'dataset_replay = openarm_data_collection.dataset_replay:main',
            'quest_arm_bridge = openarm_data_collection.quest_arm_bridge:main',
        ],
    },
)
