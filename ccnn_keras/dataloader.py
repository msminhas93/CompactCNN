#Create data loader

from keras.preprocessing.image import ImageDataGenerator



def segmentation_loader(train_directory,test_directory,batch_size=16):
    # we create two instances with the same arguments
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    # 
    seed=1
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_directory, classes= ['Image'],
        class_mode=None,shuffle=False,batch_size=batch_size,
        seed=seed, target_size=(512,512),color_mode = 'grayscale')

    mask_generator = mask_datagen.flow_from_directory(
        train_directory, classes = ['Label'],
        class_mode=None,shuffle=False,batch_size=batch_size,
        seed=seed,target_size=(128,128),color_mode = 'grayscale')

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)




    testimage_datagen = ImageDataGenerator(**data_gen_args)
    testmask_datagen = ImageDataGenerator(**data_gen_args)

    testimage_generator = testimage_datagen.flow_from_directory(
        test_directory,classes = ['Image'],
        class_mode=None,shuffle=False,
        seed=seed,target_size=(512,512),color_mode = 'grayscale')

    testmask_generator = testmask_datagen.flow_from_directory(
        test_directory,classes= ['Label'],
        class_mode=None,shuffle=False,
        seed=seed,target_size=(128,128),color_mode = 'grayscale')

    # combine generators into one which yields image and masks
    validation_generator = zip(testimage_generator, testmask_generator)
    train_num_classes = len(image_generator.classes)
    test_num_classes = len(testimage_generator.classes)
    return train_generator,validation_generator,train_num_classes,test_num_classes





def classification_loader(train_directory,test_directory,batch_size=16):

    # we create two instances with the same arguments
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,                     
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)

    train_generator = image_datagen.flow_from_directory(
        train_directory,class_mode='binary',batch_size=batch_size,classes=['NonDefective','Defective'],
        target_size=(512,512),color_mode = 'grayscale')

    testimage_datagen = ImageDataGenerator(**data_gen_args)

    validation_generator = testimage_datagen.flow_from_directory(
        test_directory,class_mode='binary',batch_size=batch_size,classes=['NonDefective','Defective'],
        target_size=(512,512),color_mode = 'grayscale')
    train_num_classes = len(train_generator.classes)
    test_num_classes = len(validation_generator.classes)
    return train_generator,validation_generator,train_num_classes,test_num_classes