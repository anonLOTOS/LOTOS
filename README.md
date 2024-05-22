## LOTOS

This code repo contains the code for Layer-wise Orthogonalization for Training Robust Ensembles (LOTOS).

### Training ensembles:

For training an ensemble in different settings you can use `main.py` and use its arguments to define the setting of interest. For example to train an ensebmel of $3$ ResNet18 models with no batch normalization layer on cifar-10, in which each layer's spectral norm is clipped to 1 and LOTOS is used for training with $mal=0.8$ and $\lambda = 0.05$:

```python main.py --method clip --mode noBN --seed 1 --convsn 1.0 --conv_factor 0.05 --bottom_clip 1.0  --arch ResNet18  --dataset cifar --efe 0 --num-models 3 --tech vanilla```


As another example, to train $3$ ResNet18 models on cifar-100 with concatenation of conv layers and batch norm layers clipped to $1$ and using LOTOS with $mal=0.8$ and $\lambda=0.01$ for both convolutional layers and composition of conv layers and batch norm together with TRS method, the command below can be used:


`python main.py --method clip --mode wBN --seed 1 --convsn 1.0 --conv_factor 0.01 --cat_factor 0.01 --bottom_clip 0.8  --cat_bottom_clip 0.8 --arch ResNet18  --dataset cifar100 --efe 0 --num-models 3 --tech trs --cat`

This code automatically measures the transferability rate among the models of the ensemble every $20$ epochs along with the robust accuracy and accuracy of the individual models, and saves the results into a csv file.


You can used `--help` to see the details of the arguments.


### Robustness against black-box attacks:

To evaluate the robustness of an ensemble against black-box attacks, you can use `black_box.py` script. It gets the address of the surrogate model along with the address of the target ensemble. As an example, you can evaluate the transferability rate of adversarial exmaples that are generated on original ResNet18 models without batch norm on an ensemble that was trained using LOTOS using the following command:


```
python black_box.py logs/cifar_ResNet18/vanilla_orig_noBN_1 --base_classifier_2 logs/cifar_ResNet18/vanilla_ortho_convonlyFalse_1stconvFalse_efe0_cat50_conv1_catclip1.0_mal1.0_mcat1.0_convfac0.05_catfac0.0_noBN_1 --method orig --mode noBN --attack_type pgd --choice best --seed 1 --num-models 3 --dataset cifar --model ResNet18
```

For further detail about the arguments, please use `--help` argument.


