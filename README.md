# Multiresolution Feature Encoding on Meshes for Implicit Functions

## Project
Implicitly learn the texture or the colors of an object as a function through the parameters of a neural network. We work with meshes as a data structure to represent our object. We propose a novel encoding method that stores features directly on the vertices of our input mesh and learns them jointly with the weights of a Multi Layer Perceptron (MLP), which decodes these features.
Our main contribution is this novel encoding method, which helps us to overcome spectral bias of neural networks. By reducing the problem of onto each face of the mesh, we are invariant to the complexity of the object and gain a sense of locality.

## Pipeline

We build an end-to-end pipeline to reconstruct the texture of our objects from images. Thus, we take our (uncolored) mesh, images of this mesh and their respective intrinsic and extrinsic camera matrix as an input and return fully rendered images of our mesh with reconstructed texture. 

## Project structure
This is the *desired* project structure, which I have adapted from [here](https://drivendata.github.io/cookiecutter-data-science/)

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── synthetic      <- Synthetically generated data.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py

```

## Data
The data for our experiments can be downloaded using the following command
```bash
./scripts/download_data.sh
```

## Installation 


## Data preprocessing

## Training 

## Inference

## Visualization
