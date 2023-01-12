from setuptools import setup, find_packages, find_namespace_packages
setup(
    name = 'causalai',
    packages=find_namespace_packages(include="*"),
    install_requires=["numpy>=1.22.2",
                    "matplotlib",
                    "scikit-learn>=1.1.2", # "scikit-learn==0.24.2",
                    "scipy>=1.4.1",
                    "ray>=1.12.0",
                    "networkx>=2.4",
                    "lingam==1.5.5",
                    "flask>=2.2.2",
                    "flask_cors>=3.0.0",
                    # "plotly==5.9.0",
                    # "dash==2.5.1",
                    # "dash-core-components==2.0.0",
                    # "dash-cytoscape==0.3.0",
                    # "dash-html-components==2.0.0",
                    # "dash-table==5.0.0"
                    ]
)

# from setuptools import setup, find_packages
# setup(
#     name = 'causallite',
#     packages=setuptools.find_packages()
# )