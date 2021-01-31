import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'C:\\Users\\guysh\\Desktop\\image_recognition\\images\\crc\\trail'
MSI = '\\MSI'
MSS = '\\MSS'

os.makedirs(root_dir +'\\train' + MSI)
os.makedirs(root_dir +'\\train' + MSS)
os.makedirs(root_dir +'\\val' + MSI)
os.makedirs(root_dir +'\\val' + MSS)
os.makedirs(root_dir +'\\test' + MSI)
os.makedirs(root_dir +'\\test' + MSS)

classifications = [MSI,MSS]

for currentCls in classifications:
    # Creating partitions of the data after shuffeling
    src = 'C:\\Users\\guysh\\Desktop\\image_recognition\\images\\crc\\train'+currentCls # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    #train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
    #                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
    train_FileNames, val_FileNames, test_FileNames, rest = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)*0.08), int(len(allFileNames)*0.095), int(len(allFileNames)*0.1)])

    train_FileNames = [src+'\\'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'\\' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'\\' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir+"\\train"+currentCls)

    for name in val_FileNames:
        shutil.copy(name, root_dir+"\\val"+currentCls)

    for name in test_FileNames:
        shutil.copy(name, root_dir+"\\test"+currentCls)