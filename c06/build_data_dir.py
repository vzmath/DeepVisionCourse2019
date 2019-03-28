import os, shutil

original_data_dir = 'data/original_cat_vs_dog_dataset'
base_dir = 'data/cat_vs_dog_small'

# make base_dir
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# in base_dir, make three sub-dirs, namely train, validation, test
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# in each of train, validation, test dirs, make two dirs cats and dogs
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)

# copy the first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(index) for index in range(1000)]
for fname in fnames:
    src_image = os.path.join(original_data_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src_image, dst)

# copy the next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(index) for index in range(1000, 1500)]
for fname in fnames:
    src_image = os.path.join(original_data_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src_image, dst)

# copy the next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(index) for index in range(1500, 2000)]
for fname in fnames:
    src_image = os.path.join(original_data_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src_image, dst)

# similar copy operations are applied to the train_dogs_dir, validation_dogs_dir, and test_dogs_dir
# copy the first 1000 dog images to train_cats_dir
fnames = ['dog.{}.jpg'.format(index) for index in range(1000)]
for fname in fnames:
    src_image = os.path.join(original_data_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src_image, dst)

# copy the next 500 cat images to validation_cats_dir
fnames = ['dog.{}.jpg'.format(index) for index in range(1000, 1500)]
for fname in fnames:
    src_image = os.path.join(original_data_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src_image, dst)

# copy the next 500 cat images to test_cats_dir
fnames = ['dog.{}.jpg'.format(index) for index in range(1500, 2000)]
for fname in fnames:
    src_image = os.path.join(original_data_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src_image, dst)

# check how many images are in each train/val/test split
#print('total training cat images:', len(os.listdir(train_cats_dir)))
#print('total validation cat images:', len(os.listdir(validation_cats_dir)))
#print('total test cat images:', len(os.listdir(test_cats_dir)))
#print('total training dog images:', len(os.listdir(train_dogs_dir)))
#print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
#print('total test dog images:', len(os.listdir(test_dogs_dir)))
