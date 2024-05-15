from setuptools import setup, find_packages


# Setting up
setup(
        name="f1tenth", 
        version='0.0.1',
        author="Luca Albinescu",
        author_email="<luca.albinescu@gmail.com>",
        description="Combining Imitation and Reinforcement Learning to Surpass Human Performance",
        packages=find_packages(),
        
        keywords=['python', 'autonomous racing'],
        classifiers= [
            "Purpose :: Dissertation",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS",
        ]
)
