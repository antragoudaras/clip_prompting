
## Installation

[`clip`](https://github.com/openai/CLIP) is needed for this project. You can install it with:

```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Overview
CLIP (Contrastive Language-Image Pre-Training) by OpenAI is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the in-context learning capabilities of GPT-2 and 3.

The datasets used are `CIFAR-10` & `CIFAR-100`. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The CIFAR-100 dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class.

We use CLIP for the following sets of taks:

## Zero-shot Image Classification
Prompting, recently popular in Natural Language Processing, involves adapting large language models for new tasks. Unlike traditional transfer learning methods that adjust neural weights, prompting transforms data directly to achieve the desired objective, diverging from the model's original training. Notably, in this process, the model parameters remain static, avoiding updates through back-propagation during training. This static parameter approach efficiently reduces trainable parameters when adjusting the model to a new task. Thus, prompting offers an attractive solution for leveraging pre-trained models, enabling users to address challenges more effectively compared to using smaller, task-specific models, even without the resources for model training.

**Key steps**

1.  Class `ZeroshotCLIP`.
    * `precompute_text_features()`: This should compute CLIP features for each class label in the dataset. Please see the function for more details.
    * `model_inference()`: This function takes an image and returns logit scores for all classes in the dataset. Please see the function for more details.
2. `main()` function -> Inference Loop.

**Running the code**

```shell
# Run the script
python $code_dir/clipzs.py --dataset cifar10 --split test
```
This will run zero-shot evaluation of CLIP on test set of CIFAR-10 dataset.

* If you want to run it through a jobscript, a sample is provided:
    ```sh
    sbatch run_clipzs.sh
    ```
    To run variants, please see the arguments in `clipzs.py` and change the arguments accordingly.

## Text Prompting with CLIP
One can supply new text prompts to CLIP to make predictions for unseen classes in the data, and visualize the result:
```shell
dataset=cifar100 # cifar10, cifar100
split=test # train, test
prompt_template='The image contains mostly the color {}'
classes='red blue green'
python clipzs.py --dataset $dataset --split $split --prompt_template "$prompt_template" \
 --class_names $classes --visualize_predictions
```


## Visual Prompting with CLIP

The goal of visual prompting is to modify the input (image) in a way that helps steer the model to perform better at a given task, e.g. image classification.

In this case, our modification of the image is *additive*, i.e., for a given input image $x$, we modify it as follows: $x' = x + v$
where $v$ is a (learnable) variable that is the *same* for each image in the dataset, and specific to a given task. It is easier to think of $v$ as a (frame) padding over the image as shown in the Fig below. This approach is called *visual prompting*.

<img width="800" alt="image" src=images/visual_prompting.png>

**Visual prompts**

There can be various ways of defining $v$, e.g., padding around the image. We implement the following: 

1. *Single pixel patch*: Class `FixedPatchPrompter` in `vp.py`. Note that the patch size is a parameter to this class. See (a) in figure below.
2. *Random pixel patch*: Class `RandomPatchPrompter` in `vp.py`. Note that the patch size is a parameter to this class. See (b) in figure below.
3. *Padding*: Class `PadPrompter` in `vp.py`. Note that the padding size is a parameter to this class. See (c) in figure below.

<img width="800" alt="image" src="https://user-images.githubusercontent.com/8458550/201727624-389295ac-8fa6-4e95-8972-5297f3191010.png">

**Visual Prompt Tuning (VPT) Model**

`vpt_model.py`. Specifically, the following functions, are implemented:
1. In the `__init__()` function, we pre-compute text features.
2. Then add the `__forward__()` function which does the following:
    * First, attaches the visual prompt to the input image
    * Then, does the usual forward pass of CLIP

**VPT Training**

The visual prompters and the model read is implemented in `learner.py` script.

**Running the Code**

Running `main.py` will start the training process. You can use `--help` to see the available options.
A sample job script (`run_clipvp.job`) is provided that runs this part with default arguments for CIFAR-10.

## Deep Prompting with CLIP
Up to this point, visual prompts have been integrated into the model's input within the image domain. Another method involves diverging from the image domain and constructing what are known as deep prompts, which are included as tokens within an intermediate representation at a specific layer of the model. Essentially, instead of incorporating prompts directly into an image, they are affixed to the original token list derived from the input. In the context of CLIP, these deep tokens can be inserted prior to any Transformer block. Refer to Figure below for an illustration of the notion of deep prompts for Transformer-based vision models.

<img width="800" alt="image" src=images/deep_prompting.png>

`dpt_model.py`. Specifically, the following functions, are implemented:
1. In the `__init__()` function, we pre-compute text features.
2. Then add the `__forward__()` function which does the following:
    * inject learnable patches in the specified injection layer
    * Then, does the usual forward pass of CLIP

## Robustness to Noise

The robustness of your learnt prompts can be evaluated against distributional shifts. To do this, one can add Gaussian noise to the test set of each dataset and observe whether there is a significant drop in performance.

1. Function ``__call__()`` of the class ``AddGaussianNoise`` in the file ``dataset.py``:
    - This function adds a Gaussian noise $𝒩(\mu, \sigma^2)$ to a batch of images.

2. robustness.py with the argument ``--test_noise`` to add noise to the test set’s images, and the argument ``--resume`` to load the best performant checkpoint as:

```shell
python robustness.py --dataset {cifar10/cifar100} --resume ./{path_to_checkpoint_folder}/model_best.pth.tar --evaluate --test_noise
```

You also can evaluate your model performance on without noise with:
```shell
python robustness.py --dataset {cifar10/cifar100} --resume ./{path_to_checkpoint_folder}/model_best.pth.tar --evaluate
```


## Cross-dataset Evaluation

In this section, one can evaluate the effectiveness of the learnt visual prompts on the combination of CIFAR-10 and CIFAR-100. More specifically, we compare CLIP’s performance when predicting a class out of both datasets’ labels (i.e. perform a 110-way classification, as the two sets of classes are mutually exclusive) with its performance on each dataset individually from the previous questions.

1. In ``cross_dataset.py`` includes the main functions needed for testing the effectiveness in this cross-dataset case.
2. Call cross_dataset.py with argument ``--evaluate`` for evaluation mode, and argument ``--resume`` to load the best performing checkpoint as:

```shell
python cross_dataset.py --dataset cifar10 --resume ./{path_to_checkpoint_folder}/model_best.pth.tar --evaluate
```
```shell
python cross_dataset.py --dataset cifar100 --resume ./{path_to_checkpoint_folder}/model_best.pth.tar --evaluate
```



