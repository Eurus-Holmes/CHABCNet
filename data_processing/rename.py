import os

path = 'images/'
fileList = os.listdir(path)

for oldName in fileList:
    print(oldName)
    newName = path + oldName.split('.')[0].split('_')[-1] + '.jpg'
    print(newName)
    os.rename(path+oldName, newName)
    print(oldName, '======>', newName)
