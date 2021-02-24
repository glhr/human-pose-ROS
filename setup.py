## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup


try:
    from catkin_pkg.python_setup import generate_distutils_setup
    # fetch values from package.xml
    setup_args = generate_distutils_setup(
        packages=['open_pose','eval','pose_utils'],
        package_dir={'': 'src'})

    setup(**setup_args)
except ModuleNotFoundError:
    setup(
        name="human-pose-ros",
        packages=['open_pose','eval','pose_utils'],
        package_dir={'': 'src'}
    )
