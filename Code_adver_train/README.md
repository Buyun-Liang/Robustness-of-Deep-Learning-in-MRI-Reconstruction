# MRI Reconstruction with Adversarial Training
This is project code for CSCI 8980 Think Deep Learning. The goal is to improve the stability of deep learning model on MRI reconstruction with adversarial training.

## Dependencies and Installation
To run this project, it's required to install fastMRI github packages to perform some pre-processing.
Refer to https://github.com/facebookresearch/fastMRI on "Dependencise and Installation" section about how to setup the environment. After installing the environment, put the Code_adver_train directory to ./FastMRI/fastMRI-master.

## Files Description
final_report.pdf is the final report of the project. FastMriDataModule.py is the data pre-processing code from FastMRI. generator.py is the generator model. unet_model.py is unet model(reconstructor) we used. PerformanceMetrics.py is performance metrics we used. train.py is the main training function we used. train_pert.py is the training for perturbation alone. pytorch_ssim directory is pytorch differntiable SSIM from https://github.com/Po-Hsun-Su/pytorch-ssim.

To see how we visualize the MRI images with and without perturbation, check Visualization.ipynb. To run the notebook, trained model need to be loaded with pytorch.

## Model Training
To train generator and reconstructor and store the models, run 

```bash
python train.py --data_path data_path --batch_size batch_size --learning_rate learning_rate --mask_type mask_type --center_fractions center_fractions --accelerations accelerations --alpha_1 alpha_1 --alpha_2 alpha_2 --epsilon epsilon
```

To run the model in Minnesota Super Institute(MSI), extra environment setting for cuda is needed. For example, in my MSI account, the environment can be set as

```bash
source activate fastmri;
module load cuda cuda-sdk;
module load python3/3.8.3_anaconda2020.07_mamba;
deviceQuery | grep NumDevs;
export LD_LIBRARY_PATH=/home/csci5980/huan1780/.conda/envs/fastmri/lib:$LD_LIBRARY_PATH;
```
where the group(csci5980) and user(huan1780) directory need to be changed to your own directory name.