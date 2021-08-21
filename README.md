# Animesion
An end-to-end framework for anime character recognition and tagging.
![](./classification_tagging/data_exploration/figures/AnimesionSystemDiagram.png)

Install requirements by first creating a conda environment and installing with conda then pip
packages for packages that cant be found in conda
```
conda create --name animesion --file requirements_conda.txt
conda activate animesion
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
cd classification_tagging/models & pip install -e . & python download_convert_models.py
```
