# Playground for AutoEncoder


## Code Architecture

- `model.py`: model architecture classes will be written here
- `training.py`: training loop
- `testing.py`: testing loop
- `inference.py`: inference loop
- `config.yaml`:
- `utils.py`: 
- `run.sh`: having all the run commands
- `data/`: having all the data files
- `requirements.txt`: 


## How to Run

`python main.py --model SimpleVAE --unittest`


## Applications




## Plan


### 2

- make a vae model, (done)
- use a random fixed tensor to verify model, (done)


### 1

- make a basic autoencoder model
- run it through 28x28 images
- load the mnist dataset
- make the model work properly with the image
- make inference function to check the effectiveness of image reconstruction

- try lowering the test reconstruction error,
- try lowering the dimension of the latent space,
- try using a probabilistic latent space,

- do pca on latent space,

