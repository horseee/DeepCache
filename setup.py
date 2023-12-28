import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepCache",
    version="v0.1.0",
    author="Xinyin Ma",
    author_email="maxinyin@u.nus.edu",
    description="DeepCache: Accelerating Diffusion Models for Free",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/horseee/DeepCache",
    packages=setuptools.find_packages(exclude=["DeepCache.ddpm"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=['torch', 'diffusers', 'transformers'],
    python_requires='>=3.6',
)