import subprocess
import sys
import importlib.util
import os

def is_installed(package_name):
    # Try to find the module and return whether it's found
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None

def install(package):
    if is_installed(package):
        print(f"{package} is already installed.")
    else:
        # If not installed, install the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} has been installed.")

def run_git_commands():
    if is_installed("pymobility"):
        print("pymobility is already installed.")
    else:
        # Clone the repository
        subprocess.check_call(["git", "clone", "https://github.com/panisson/pymobility.git"])
        # Change directory
        os.chdir("pymobility")
        # Install the package using setup.py
        subprocess.check_call([sys.executable, "setup.py", "install"])

        print("pymobility has been cloned and installed")

# Mapping of package names to their standard import names if they differ
package_to_import_name = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "geopy": "geopy",
    "math": "math",
    "folium": "folium",
    "scikit-learn": "scikit-learn",
    "simpy": "simpy",
    "tqdm": "tqdm",
    "imageio": "imageio",
    "nbformat": "nbformat",
    "geopandas": "geopandas",
    "dill":"dill",
    "wdbo_algo": "wdbo_algo",
    "wdbo_criterion": "wdbo_criterion",
    "imageio[pyav]": "imageio[pyav]",
    "imageio[opencv]": "imageio[opencv]",
    "selenium":"selenium",
    "multiprocess":"multiprocess",
    "scipy":"scipy"
}

if __name__ == "__main__":
    for package, import_name in package_to_import_name.items():
        install(import_name)
    print("All packages are up to date.")

    # Run git commands
    run_git_commands()

