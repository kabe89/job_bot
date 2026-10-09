from setuptools import setup, find_packages

setup(
    name="jobbot",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"jobbot": ["templates/*.html", "static/*.css"]},
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={"console_scripts": ["jobbot=jobbot.cli:main"]},
    python_requires=">=3.10",
)
