base_architecture = 'vgg19'
img_size = 224
prototype_shape = (30, 128, 1, 1) #10 para cada classe #60
num_classes = 3 #6
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '003'

data_path = '/home/up202003072/Documents/ProtoPNet/datasets/bone_cropped/'
train_dir = data_path + 'train_cropped_augmented/' #augmented e balanded
test_dir = data_path + 'testing/'
train_push_dir = data_path + 'training_new/' #unbalenced e original 
# train_batch_size = 80 #26
# test_batch_size = 80 #26
# train_push_batch_size = 75 #20
train_batch_size = 1 #15
test_batch_size = 1 #15
train_push_batch_size = 1 #20

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-3 

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}


num_train_epochs = 85 #train epochs #70
num_warm_epochs = 0 #number of warm-up epochs #5

push_start = 42 #10
push_epochs = [42, 82]


