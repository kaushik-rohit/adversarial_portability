# adversarial_portability
Study on portability of targeted and untargeted adversarial attacks. We select 13 different neural network architecture
and generate adversarial examples using FGSM, PGD and FGV with different pertubation constant of 0.01, 0.1, 0.3

# generate adversarial examples

The scripts create_adversarial_ex and create_adversarial_ex_targeted are used to generate the adversarial images
on our dataset. The batch size, attacks and gpu_ids can be defined as parameter in the script
