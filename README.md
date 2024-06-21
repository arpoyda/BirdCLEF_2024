Below you can find an outline of how to reproduce [our solution](https://www.kaggle.com/competitions/birdclef-2024/discussion/512197) for the BirdCLEF 2024 competition.


## Hardware/Software:

Kaggle kernel with Tesla P100.

Python packages are detailed separately in `requirements.txt`


## Data Setup

Competition data can be downloaded from [here](https://www.kaggle.com/competitions/birdclef-2024/data). Put it in the `./data/` directory.


## Model Build

There are three options to produce the solution.
1) With pretrained models.
    
    a) Download models checkpoints from [here](https://www.kaggle.com/datasets/chemrovkirill/birdclef24-final) and put in `models_weights/`.

    b) Run `python run_inference.py` from the top directory.

    Submission file will be in `tmp/` directory.

2) Retrain from sratch.

    Run the following scripts from the top directory.
    
    a) Create mels and pseudo labeles for `data/train_audio`:
    
       `python run_train_preprocessing.py`
    
    b) Create mels and pseudo labeles for `data/unlabeled_soundscapes`:
    
       `python run_unlabeled_preprocessing.py` 
    
    c) Train 6 models (for each config from `configs/` in a similar way):

       `python run_training.py --cfg configs/effnet_seg20_80low.json`
    
    d) Make predicttion for `data/test_soundscapes`:
    
       `python run_inference.py`

    Submission file will be in `tmp/` directory.


## Inference on Kaggle

Inference is published in a Kaggle kernel [here](https://www.kaggle.com/code/chemrovkirill/birdclef-2024-1st-place-inference). Weights from our trained models are provided in a kaggle dataset linked to the inference kernel [here](https://www.kaggle.com/datasets/chemrovkirill/birdclef24-final).

Also see [our write-up](https://www.kaggle.com/competitions/birdclef-2024/discussion/512197) for more details.
