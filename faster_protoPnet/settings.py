base_architecture = 'vgg19'
img_size = 800
prototype_shape = (40, 128, 1, 1) #10 para cada classe #60
num_classes = 4 #6
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '003'

data_path = '/home/up202003072/Documents/ProtoPNet/datasets/bone_cropped/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'testing/'
train_push_dir = data_path + 'training/'
train_batch_size = 80
test_batch_size = 80
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 70 #train epochs
num_warm_epochs = 5

push_start = 10
#This variable indicates the epoch number at which prototype pushing begins. 
# Prototype pushing is a technique where the prototype vectors are adjusted based on the examples from the push set. 
# It helps to refine the prototypes and improve the model's performance.

push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]