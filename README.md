#  Audio-Visua Lip Synthesis via intermediate landmark representation |  Final Year Project (Dissertation) of Wish Suharitdamrong


This is a code implementation for Wish Suharitdamrong's Final Year Project Year 3 BSc Computer Science at University of Surrey on the topic of Audio-Visua Lip Synthesis via intermediate landmark representation.


![Alt Text](./dummy/face.gif)

# Demo

Online demonstration is available at 🤗 [HuggingFace](https://huggingface.co/spaces/peterwisu/lip_synthesis)

## Installation


There are two ways of installing package using conda or pip

1.Create virtual conda environment from `environment.yml`

2.Use pip to install a pakages (make sure you use `python 3.7`or above since older version might not support some libraries)

### Use Conda

```bash
# Create virtual environment from .yml file
conda env create -f environment.yml

# activate virtual environment
conda activate fyp
```

### Use pip


```bash
# Use pip to install require packages
pip install -r requirement.txt
```

## Dataset

The audio-visual dataset used in this proejct are LRS2 and LRS3. LRS2 data was use for both model training and evaluation. LRS3 data was only used for model evaluation.

| Dataset                                 |  Page|  
|----------                             |:-------------:|
| LRS2                            |  [Link](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)|
| LRS3                 |    [Link](https://paperswithcode.com/dataset/lrs3-ted) | 


## Pre-train weights



### Generator model 
Download weights Generator model
| Model                                 |  Donwload Link  |  
|----------                             |:-------------:|
| Generator                             |  [Link](https://drive.google.com/file/d/19-zLzCKeH6tp5grxoRYnEEKLgIZrj-4f/view?usp=sharing)|
| Generator + SyncLoss                  |    [Link](https://drive.google.com/file/d/1Ck-54fOBeY87c6_CFXfMF0FwgWK92DqG/view?usp=sharing) | 
| Attention Generator + SyncLoss   | [Link](https://drive.google.com/file/d/1sEM7Aqrg-YILx8dyuT2zxQkU5xRJXc_T/view?usp=sharing) |

### Landmark SyncNet discriminator


Download weights for Landmark-based SyncNet model [Download Link](https://drive.google.com/file/d/1fJj-zYkfr1gSGgq5ISWGCE1byxNc6Mdp/view?usp=sharing)

### Image-to-Image Translation

Pre-trained weight for Image2Image Translation model can be download from MakeItTalk repository on their pre-trained models section [Repo Link](https://github.com/yzhou359/MakeItTalk).

### Directory
```bash
├── checkpoint #  Directory for model checkpoint
│   └── generator   # put Generator model weights here
│   └── syncnet     # put Landmark SyncNet model weights here
│   └── image2image # put Image2Image Translation model weights here
```

## Run Inference

```
python run_inference.py --generator_checkpoint <checkpoint_path> --image2image_checkpoint <checkpoint_path> --input_face <image/video_path> --input_audio <audio_source_path>
```

## Data Preprocessing 

I used same ways of data preprocessing as [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) for more details of folder structure can be find in their repository [Here](https://github.com/Rudrabha/Wav2Lip).

```
python preprocess_data.py --data_root data_root/main --preprocessed_root preprocessed_lrs2_landmark/
```

## Train Model 

### Generator 


```
# CLI for traning attention generator with pretrain landmark SyncNet discriminator
python run_train_generator.py --model_type attnlstm --train_type pretrain --data_root preprocessed_lrs2_landmark/ --checkpoint_dir <folder_to_save_checkpoints>
```

### Landmark SyncNet


```
# CLI for training pretrain landmark SyncNet discriminator
python run_train_syncnet.py --data_root preprocessed_lrs2_landmark/ --checkpoint_dir <folder_to_save_checkpoints>
```

## Generate video for evaluation & benchmark from LRS2 and LRS3

This project used data from LRS2 and LRS3 dataset for quantitative evaluation, the list of evaluation data is provide from [Wav2Lip](https://github.com/Rudrabha/Wav2Lip). The filelist(video and audio data used for evaluation) and details about Lip Sync benchmark are available in their repository [Here](https://github.com/Rudrabha/Wav2Lip). 

### Generate evaluation from filelist
```
cd evaluation
# generate evaluation videos
python gen_eval_vdo.py --filelist <path> --data_root <path>  --model_type <type_of_model> --result_dir <save_path> --generator_checkpoint <gen_ckpt> --image2image_checkpoint <image2image_checkpoint>
```




# Acknowledgement 


The code base of this project was inspired from [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) and [MakeItTalk](https://github.com/yzhou359/MakeItTalk). I would like to thanks author of both project for making code implementation of their amazing work available online. 




