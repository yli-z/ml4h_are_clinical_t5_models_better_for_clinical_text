# Are Clinical T5 Models Better for Clinical Text?

This is the official code repository for the ML4H(2024) paper "Are Clinical T5 Models Better for Clinical Text?" 

## Abstract
Large language models with a transformer-based encoder/decoder architecture, such as T5, have become standard platforms for supervised tasks. To bring these technologies to the clinical domain, recent work has trained new or adapted existing models to clinical data. However, the evaluation of these clinical T5 models and comparison to other models has been limited. Are the clinical T5 models better choices than FLAN-tuned generic T5 models? Do they generalize better to new clinical domains that differ from the training sets? We comprehensively evaluate these models across several clinical tasks and domains. We find that clinical T5 models provide marginal improvements over existing models, and perform worse when evaluated on different domains. Our results inform future choices in developing clinical LLMs.

## Data and Models

We do not release any new data or models as part of this study. Instead, we utilize and evaluate existing resources:

### Pre-trained Models
* Clinical-T5(Lehman et al., 2023): [Model](https://physionet.org/content/clinical-t5/1.0.0/) | [Paper](https://arxiv.org/abs/2302.08091)
* Clinical-T5(Lu et al., 2022): [Model](https://huggingface.co/luqh/ClinicalT5-large) | [Paper](https://aclanthology.org/2022.findings-emnlp.398/)
* Den-T5 ([Model](https://huggingface.co/google/t5-v1_1-large)) and Sup-T5 ([Model](https://huggingface.co/google-t5/t5-large)): [Paper](https://arxiv.org/abs/1910.10683)
* FLAN-T5: [Model](https://huggingface.co/google/flan-t5-large) | [Paper](https://arxiv.org/abs/2210.11416)

### Datasets
* Clinical Datasets: [MedNLI](https://physionet.org/content/mednli/1.0.0/), [RadQA](https://physionet.org/content/radqa/1.0.0/), [CLIP](https://physionet.org/content/mimic-iii-clinical-action/1.0.0/), [Stigmatizing_data](https://physionet.org/content/stigmatizing-language/1.0.0/)
* Biomedical Datasets: [HOC](https://huggingface.co/datasets/bigbio/hallmarks_of_cancer), [NCBI_disease](https://github.com/justinphan3110/SciFive/tree/main/biot5x/data/NCBI_disease), [BC5CDR_disease](https://github.com/justinphan3110/SciFive/tree/main/biot5x/data/BC5CDR_disease)

For detailed data processing, please refer to [Clinical-T5(Lehman et al., 2023)](https://github.com/elehman16/do-we-still-need-clinical-lms) and [SciFive(Phan et al., 2021)](https://github.com/justinphan3110/SciFive) repositories.

## Implementation

### Data Preprocess
For 5-fold cross-validation experiments, firstly, we merged the training and validation splits. We then shuffled the concatenated data and splitted it into 5 folds. 

For downsampled-data experiments, we downsampled the training data within each fold. 

### Finetuning
Our finetuning experiments are based on [Clinical-T5(Lehman et al., 2023)](https://github.com/elehman16/do-we-still-need-clinical-lms) and [SciFive(Phan et al., 2021)](https://github.com/justinphan3110/SciFive). See the `src` and `scripts` directories. 

If you have questions or issues when working with this repository, please [email us](mailto:yahanli@usc.edu).

