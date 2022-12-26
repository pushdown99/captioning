**********************
CUDA TF2
**********************
python -m venv tf2
source tf2/bin/activate
pyenv install 3.9.10
python -m pip install --upgrade pip
pip install tensorflow colorama easydict
pip install tqdm ipykernel nltk

python -m ipykernel install --user --name tf2 --display-name "tf2"

pip install pipreqs
python -m  pipreqs.pipreqs . --force
pip freeze > requirements.txt

pip install -r requirements.txt
