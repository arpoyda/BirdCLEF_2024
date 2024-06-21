Below you can find a outline of how to reproduce [our solution](https://www.kaggle.com/competitions/birdclef-2024/discussion/512197) for the BirdCLEF 2024 competition.


#HARDWARE and SOFTWARE:
Kaggle kernel with Tesla P100
python packages are detailed separately in `requirements.txt`


#DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
Competition data can be downloaded from [here](https://www.kaggle.com/competitions/birdclef-2024/data). Put it in the `./data/` directory.


#MODEL BUILD: There are three options to produce the solution.
1) With pretrained models.
    a) Run `python run_inference.py` from the top directory.
    b) Submission file will be in `tmp/` directory.
3) Retrain from sratch.
    Run the following scripts from the top directory.
    a) `python run_train_preprocessing.py` to create mels and pseudo labeles for `train_audio`
    b) `python run_unlabeled_preprocessing.py` to create mels and pseudo labeles for `unlabeled_soundscapes`
    c) For each config from `configs/`:
       `python run_training.py --cfg configs/effnet_seg20_80low.json`
    d) `python run_inference.py`


#INFERENCE on Kaggle
Inference is published in a kaggle kernel [here](https://www.kaggle.com/code/chemrovkirill/birdclef-2024-1st-place-inference). Weights from our trained models are provided in a kaggle dataset linked to the inference kernel [here](https://www.kaggle.com/datasets/chemrovkirill/birdclef24-final).

Also see [our write-up](https://www.kaggle.com/competitions/birdclef-2024/discussion/512197) for more details.
