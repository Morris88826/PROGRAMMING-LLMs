# CSCE 689 - Homework 2

## Overview
In this homework, I implemented a variant of GPT-2 from scratch, based on the Python implementation from Andrej Karpathy’s [llm.c](https://github.com/karpathy/llm.c/tree/master) repository. Given the limited dataset size (fineweb10B) and training resources (A100 GPU for 24 hours), it was crucial to choose a suitable number of parameters and implement efficient techniques to ensure the model converged faster. According to the scaling laws for large language models, there is a trade-off between model size and dataset size: as one increases, the other should also increase proportionally to avoid underfitting or overfitting. Since our dataset is relatively small (10B tokens) and training time is short, I experimented with two smaller model sizes: 124M and 350M. The checkpoint corresponding to the step with the lowest validation loss was selected for evaluation. Both models were evaluated on the HellaSwag dataset, yielding normalized accuracies of <b>30.78%</b> and  <b>33.52%</b> for the 124M and 350M models, respectively.

## Model Card
* custom-gpt-124M: [here](https://huggingface.co/Morris88826/Mu-Ruei_Tseng_133007868_124M)
* custom-gpt-350M: [here](https://huggingface.co/Morris88826/Mu-Ruei_Tseng_133007868_350M)
### Training Resource
Trained on the FineWeb10B dataset (10 billion tokens) using one NVIDIA A100 GPU, with a total of 26 GPU hours (24 hours plus an additional 2 hours).

## Get Started
```
git clone https://github.com/Morris88826/CSCE-689-HW2.git
cd CSCE-689-HW2

# clone the huggingface model
git clone https://huggingface.co/Morris88826/Mu-Ruei_Tseng_133007868_124M
git clone https://huggingface.co/Morris88826/Mu-Ruei_Tseng_133007868_350M

# evaluate on hellaswag
./dev/eval/run_eval.sh Mu-Ruei_Tseng_133007868_124M log124M
./dev/eval/run_eval.sh Mu-Ruei_Tseng_133007868_350M log350M
```

## Results
We present the loss graph for the fine-tuned 10B dataset and evaluation results on HellaSwag. The custom 124M-parameter GPT-2 variant shows faster convergence and lower validation loss compared to the original GPT-2, resulting in a slight accuracy improvement from 29.4% to 30.78%. To further explore the impact of model size, a 350M-parameter version was trained. Despite completing fewer steps, it achieved a significantly lower loss and improved accuracy to 33.52%, nearing GPT-3’s 124M model performance. For more details regarding the implementation, please see [here](./report.pdf).

![output](https://github.com/user-attachments/assets/f6533c87-b202-4cc5-bb23-c16f55054d44)
