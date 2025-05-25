# Install miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# If you have problems running conda from your terminal
source ~/miniconda3/etc/profile.d/conda.sh

# Create a Python env version 3.9 (according to stack overflow it is a more stable version for this lib)
conda create -n gdal_env python=3.9 gdal notebook ipykernel -c conda-forge -y  

# Activate the env
conda activate gdal_env

# Add the env to jupyter kernel so we can use it in Visual Studio
python -m ipykernel install --user --name=gdal_env --display-name "Python (GDAL)"
