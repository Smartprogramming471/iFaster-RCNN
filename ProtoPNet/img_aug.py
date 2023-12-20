import Augmentor
import os
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

datasets_root_dir = '/home/up202003072/Documents/ProtoPNet/datasets/bone_cropped/'
dir = datasets_root_dir + 'training/'
target_dir = datasets_root_dir + 'train_cropped_augmented/'

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(1):
    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    p = Augmentor.Pipeline('/home/up202003072/Documents/ProtoPNet/datasets/bone_cropped/training')
    p.rotate_without_crop(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # skew
    p = Augmentor.Pipeline('/home/up202003072/Documents/ProtoPNet/datasets/bone_cropped/training')
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # shear
    # p = Augmentor.Pipeline('/home/up202003072/Documents/ProtoPNet/datasets/bone_cropped/training')
    # p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    # p.flip_left_right(probability=0.5)
    # for i in range(10):
    #     p.process()
    # del p
