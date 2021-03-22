# CoNAL Pytorch Implementation
Unofficial pytorch implementation of CoNAL(Common Noise Adaptation Layers) from "Learning from Crowds by Modeling Common Confusions"
- Paper: https://arxiv.org/pdf/2012.13052.pdf
- Index
  - [Preparing Task](#Preparing-Task)
  - [Training Networks](#Training-Networks)
  - [Testing Networks](#Testing-Networks)
  - [Experiments](#Experiments)  

## Preparing Task
In our implementation, there are two tasks which were used on the original paper - LabelMe and Music.
If you want to train another crowd-sourcing task, you should follow the instructions.

### Create package
First, you need to create a new python package. It is done by copying one of the existing task package.
```bash
cp -r tasks/music tasks/<your-task>
```
Then your task package will have the following structure.
```bash
tasks/<your-task>
├── __init__.py
├── argument.py
├── classifier.py
└── dataset.py
```

### Implement dataset
In `dataset.py`, there is `Dataset` class that is used to fetch crowd-sourcing data. 
You need to implement `__init__()`, `__len__()`, and `__getitem__()`.
While implementing, ***you should not change the signature of the functions***.
For each function, there are some requirements you should meet.

- `__len__()` : return the number of data.
- `__getitem__()` : return the data which its index is `idx`, and any shape doesn't matters

If you need additional arguments for dataset, modify `add_dataset_args()`. Additional information is introduced at [argparse](https://docs.python.org/3/library/argparse.html).

### Implement classifier
In `classifier.py`, there is `Classifier` class that is used for generating the latent variable.
It is same with other pytorch models, but one restriction exists - the output shape must be `(batch_size, n_class)`.
Input variable `x` in `forward` function is the data you fetched from `Dataset` instance.
As same as `Dataset`, if you want more arguments, modify `add_classifier_args()`.

### Add to global arguments
After finishing `Dataset` and `Classifier`, please add your task to `arguments.py`.
```python
# tasks = ['labelme', 'music']
tasks = ['<your-task>', 'labelme', 'music']
```
Now you can train your own crowd-sourcing task with CoNAL!

## Training Networks
Training is simple. 
Run `python train.py --task <your-task> -h` to see the arguments and then add them.
Because the arguments for each task is different, you should add `--task` option to see help.
After adding the arguments, just run the command!
Logs will be saved as tensorboard log that you can see on tensorboard.
Checkpoints are also saved and the usage will be described on "Test Networks".

```bash
> python train.py --task labelme -h
usage: train.py [-h] [--seed SEED] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--log_interval LOG_INTERVAL] [--task {labelme,music}] [--train_data TRAIN_DATA] [--valid_data VALID_DATA] [--device {cpu,cuda}] [--save_dir SAVE_DIR] [--log_dir LOG_DIR] [--scale SCALE] [--input_dim INPUT_DIM]
                [--n_class N_CLASS] [--n_annotator N_ANNOTATOR] [--emb_dim EMB_DIM] [--dropout DROPOUT] [--n_units N_UNITS]

optional arguments:
  -h, --help            show this help message and exit

train:
  --seed SEED           Random seed.
  --epochs EPOCHS       Number of epochs for training.
  --batch_size BATCH_SIZE
                        Number of instances in a batch.
  --lr LR               Learning rate.
  --log_interval LOG_INTERVAL
                        Log interval.
  --task {labelme,music}
                        Task name for training.
  --train_data TRAIN_DATA
                        Root directory of train data.
  --valid_data VALID_DATA
                        Root directory of validation data.
  --device {cpu,cuda}   Device going to use for training.
  --save_dir SAVE_DIR   Folder going to save model checkpoints.
  --log_dir LOG_DIR     Folder going to save logs.
  --scale SCALE         Scale of regularization term.

model:
  --input_dim INPUT_DIM
                        Input dimension of CoNAL.
  --n_class N_CLASS     Number of classes for classification.
  --n_annotator N_ANNOTATOR
                        Number of annotators that labeled the data.
  --emb_dim EMB_DIM     Dimension of embedding in auxiliary network of CoNAL.

classifier:
  --dropout DROPOUT     Dropout rate
  --n_units N_UNITS     Number of units in FC layer

```

## Testing Networks
### Measure accuracy of test set
With your checkpoint, you can measure accuracy of your classifier on test data using `test.py`.
Just give your checkpoint directory and test data path, and it will automatically calcualte the total accuracy.
```bash
> python test.py -h
usage: test.py [-h] [--batch_size BATCH_SIZE] [--test_data TEST_DATA] [--device {cpu,cuda}] [--ckpt_dir CKPT_DIR]

optional arguments:
  -h, --help            show this help message and exit

test:
  --batch_size BATCH_SIZE
                        Number of instances in a batch.
  --test_data TEST_DATA
                        Root directory of test data.
  --device {cpu,cuda}   Device going to use for training.
  --ckpt_dir CKPT_DIR   Directory which contains the checkpoint and args.json.
```

### Analyze Confusion Matrices
Also in our checkpoint file, there are the weights of confusion matrices in noise adaptation layer.
You can load using the following code.
```python
import torch
from model import NoiseAdaptationLayer

checkpoint = torch.load(checkpoint_dir)
confusion_matrices = NoiseAdaptationLayer(n_class, n_annotator)
confusion_matrices.load_state_dict(checkpoint['noise_adaptation_layer'])

confusion_matrices.local_confusion_matrices  # Confusion matrices of each annotator
confusion_matrices.global_confusion_matrix   # Confusion matrix of common noise
```

## Experiments
We've held the same experiment mentioned on the original paper and compare the results.
- [Trained Models](https://postechackr-my.sharepoint.com/:f:/g/personal/shhj1998_postech_ac_kr/EiMjXRV_aaRCu2PhO5bE2hcBHyWzVLi1GtLmXtQsBT6Mpw?e=bLdvx5)
- [Preprocessed Data](https://postechackr-my.sharepoint.com/:f:/g/personal/shhj1998_postech_ac_kr/EspiotCZNIVDpNSDO7yo7iIBZhY9om-_86rjOwT5CfwSHg?e=3nCbKu)

### LabelMe
[LabelMe](http://labelme.csail.mit.edu/Release3.0/) is an image classification task that was labeled by 77 annotators in AMT(Amazon Mechanical Turk).
Original data is [here](http://fprodrigues.com/deep_LabelMe.tar.gz).
However in the real data, 18 of the annotators didn't labeled any image.
So total 59 annotators' labels were used for training.
Following the experiment setting, we trained the model in five different random seeds and pick the best one based on validation set accuracy.
- Learning Rate: 5e-3
- Batch Size: 256
- Regularization Scaling: 1e-5

We can see that it is almost same with original paper.

Model | CoNAL(paper) | CoNAL-pytorch
--- | --- | --- 
Accuracy | 87.12 ± 0.55 | 88.75 ± 0.68

### Music
[Music](http://fprodrigues.com/mturk-datasets.tar.gz) is a music genre classification task that was labeled by 44 annotators in AMT.
Following the experiment setting, we trained the model in five different random seeds and used test data as a validation data.
- Learning Rate: 1e-2
- Batch Size: 1024
- Regularization Scaling: 1e-5

It showed that our implementation is very poor compare to the original paper.
More analysis is held in future.

Model | CoNAL(paper) | CoNAL-pytorch
--- | --- | --- 
Accuracy | 84.06 ± 0.42 | 67.00 ± 0.762

## TODO
- [ ] Multi GPU training
- [ ] Weight initializations (e.g xavier, kaiming)
- [ ] Keep constraints of confusion matrices while training
- [ ] Reason of Poor Accuracy on Music Task
