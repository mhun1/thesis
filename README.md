# Thesis code

The following repository contains the code used for my thesis.

## Usage

The following command executes the training process consisting of:

1. Preprocessing
2. Augmentation
3. 3D-UNet Model
4. FP16 training & FP32 k-fold Validation (3-fold) 
5. Simultaneous logging to W&B

`python3 train.py -dataset <name> -loss <name> -dir <path_to_data> -dir_m <model_save>` 

### Requirements:


- Define your dataset via TorchIO like the examples provided in /datasets
- Define your transformations that should be used  
- Add them to the get_dataset() function
- Define the -dir with the path, where the data lays, and -dir_m with the directory, where the model should be saved
- Further dependencies can be checked in /utils/args_parser.py

### Further requirements
1. 22GB VRAM GPU needed for data allocation and model allocation
2. Listed libraries in setup.py
4. Time!



