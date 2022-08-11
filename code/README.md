---
license: apache-2.0
tags:
- image-classification
- generated_from_trainer
datasets:
- beans
metrics:
- accuracy
model-index:
- name: ''
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: beans
      type: beans
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9774436090225563
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the beans dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0666
- Accuracy: 0.9774

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.0742        | 1.54  | 100  | 0.0806          | 0.9774   |
| 0.0185        | 3.08  | 200  | 0.0666          | 0.9774   |


### Framework versions

- Transformers 4.14.1
- Pytorch 1.10.0
- Datasets 2.0.0
- Tokenizers 0.10.3
