# Deep Bayesian Active Learning-to-Rank

Official code release for the paper:

**Deep Bayesian Active Learning-to-Rank with Relative Annotation for Estimation of Ulcerative Colitis Severity**

- Medical Image Analysis, 2024
- DOI: https://doi.org/10.1016/j.media.2024.103262
- arXiv: https://arxiv.org/abs/2409.04952

## Dataset

This repository uses the public **LIMUC (Labeled Images for Ulcerative Colitis) dataset**:

https://zenodo.org/records/5827695

Please use the `patient_based_classified_images` directory included in the LIMUC dataset.

The dataset itself is not included in this repository.

## Environment

The original experiments reported in the paper were conducted using:

- Ubuntu 18.04
- TensorFlow 1.13.1
- Keras 2.2.4

For this public code release, we provide a Docker environment based on NVIDIA TensorFlow 21.05 (`nvcr.io/nvidia/tensorflow:21.05-tf2-py3`) with Keras 2.4.3.

The released code has been reorganized and updated for public use and is therefore not an exact archival copy of the original experimental code.

For the original experimental settings, please refer to the paper. Hyperparameters in this released implementation can be adjusted as needed for your environment and experimental settings.

## Directory Structure

```text
Deep-Bayesian-Active-Learning-to-Rank/
├── Docker/
│   ├── Dockerfile
│   └── run_build.sh
│
├── run_docker.sh
│
├── Data/
│   └── UC/
│       └── LIMUC/
│           └── Preprocessing/
│               ├── 1_prepare_uc_image_dataset.py
│               ├── 2_create_patient_level_splits.py
│               ├── 3_format_fold_datasets.py
│               └── 4_create_training_only_datasets.py
│
└── Experiments/
    └── Ranknet/
        └── Bayesian/
            └── LIMUC/
                └── Scripts/
                    ├── 1_initial_learning_make_pair.py
                    ├── 2_train_bayesian_ranknet.py
                    ├── 3_predict_score.py
                    ├── 4_calculate_uncertainty.py
                    ├── 5_active_learning_make_pair.py
                    ├── 6_train_bayesian_ranknet_AL.py
                    ├── bayesian_densenet.py
                    ├── callbacks.py
                    ├── test_balanced_pair_accuracy.py
                    ├── bar_graph.py
                    ├── box_plot.py
                    └── scatter_plot.py
```

Dataset directories, active-learning datasets (`Add_dataset/`), and experiment results (`Results/`) are generated during preprocessing and experiment execution.

## Usage

### 1. Prepare the LIMUC dataset

Download the LIMUC dataset from:

https://zenodo.org/records/5827695

Use the `patient_based_classified_images` directory and place it at:

```text
Data/UC/LIMUC/Images/patient_based_classified_images/
```

The preprocessing scripts below create the directory structure required for the experiments.

### 2. Build the Docker image

From the project root, run:

```bash
cd Docker
bash run_build.sh
cd ..
```

### 3. Start the Docker container

From the project root, run:

```bash
bash run_docker.sh
```

The project directory is mounted to `/workdir` inside the container.

### 4. Prepare the dataset

Inside the Docker container, move to:

```bash
cd /workdir/Data/UC/LIMUC/Preprocessing
```

Run the preprocessing scripts sequentially:

```bash
python 1_prepare_uc_image_dataset.py
python 2_create_patient_level_splits.py
python 3_format_fold_datasets.py
python 4_create_training_only_datasets.py
```

### 5. Run Bayesian active learning-to-rank

Move to:

```bash
cd /workdir/Experiments/Ranknet/Bayesian/LIMUC/Scripts
```

To start the active-learning procedure, run scripts 1–6 sequentially:

```bash
python 1_initial_learning_make_pair.py
python 2_train_bayesian_ranknet.py
python 3_predict_score.py
python 4_calculate_uncertainty.py
python 5_active_learning_make_pair.py
python 6_train_bayesian_ranknet_AL.py
```

After completing script 6, repeat scripts 3–6 for each subsequent active-learning iteration:

```bash
python 3_predict_score.py
python 4_calculate_uncertainty.py
python 5_active_learning_make_pair.py
python 6_train_bayesian_ranknet_AL.py
```

Continue this cycle until the desired active-learning iteration is reached.

In the experiments reported in the paper, the number of active-learning iterations was set to `K = 6`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{KADOTA2024103262,
  title   = {Deep Bayesian active learning-to-rank with relative annotation for estimation of ulcerative colitis severity},
  journal = {Medical Image Analysis},
  volume  = {97},
  pages   = {103262},
  year    = {2024},
  issn    = {1361-8415},
  doi     = {10.1016/j.media.2024.103262},
  url     = {https://www.sciencedirect.com/science/article/pii/S1361841524001877},
  author  = {Takeaki Kadota and Hideaki Hayashi and Ryoma Bise and Kiyohito Tanaka and Seiichi Uchida}
}
```
