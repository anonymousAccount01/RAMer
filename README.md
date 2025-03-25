# RAMer: Reconstruction-based Adversarial Model for Multi-party Multi-model Multi-label Emotion Recognition

This is an anonymous repository for double-blind manuscript review.

## TODO List

- [x] [Overview](#overview)  
  - The model framework.

- [x] [Requirements](#requirements)
  - List all dependencies and libraries required to run the code.

- [x] [Training](#training)  
  - Provide instructions on how to train the model, including command examples and parameter explanations.

- [x] [Evaluation](#evaluation)  
  - Explain the evaluation process and provide examples of how to evaluate the model's performance.

## Overview
The framework of RAMer. Given incomplete multi-modal inputs, RAMer first encodes each individual modality through an auxiliary task, then feeds the features into a reconstruction-based adversarial network to extract specificity and commonality. Finally, a stacked shuffle layer is employed to learn enhanced representations.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
sh train.sh
```

>📋  Describe the training details, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on CMU-MOSEI, M3ED and MEmoR, run:

```eval
inclued in train.sh
```

>📋  Describe the evaluation process, and give commands that produce the results.

## Models

You can download the checkpoint of models here:

- [model]. 


## Contributing

>📋  Pick a licence. 
