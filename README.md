# adversarial_portability
Study on portability of targeted and untargeted adversarial attacks. We select 13 different neural network architecture
and generate adversarial examples using FGSM, PGD and FGV with different pertubation constant of 0.01, 0.1, 0.3

# generate adversarial examples

The scripts create_adversarial_ex and create_adversarial_ex_targeted are used to generate the adversarial images
on our dataset. The batch size, attacks and gpu_ids can be defined as parameter in the script.

Our dataset is a subset of the Imagnet ILSVRC 2012 data. The data.py script is used to select 10 images
for each of the 1000 classes that are predicted correctly by all the selected neural network.

# portability
The fixed_alpha_predictions and fixed_alpha_targeted_predictions are used to study the portability of
the network. These scripts runs the evaluation on the generated adversarial examples and stores the
prediction in a csv format.

The distance metric and other statistics are calculated in the utils.py script and we include visualization
in the viz.py script
