import json
import os
from PIL import Image
import pdb


def load_img(path):
    return Image.open(path).convert('RGB')


def crop_img():
    path = r'./ReCTS/'
    label_path = 'gt_unicode/'
    img_path = 'img/'
    label_list = os.listdir(path + label_path)
    if os.path.exists(path + 'dictionary.json') == True:
        with open(path + 'dictionary.json', 'r') as f:
            dictionary = json.load(f)
    else:
        dictionary = {}
    save_path = path + '/cropped_img/'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    i = len(dictionary)
    length = len(label_list)
    for j, label in enumerate(label_list):
        img = load_img(path + img_path + label.split('.')[0] + '.jpg')
        # img.show()
        with open(path + label_path + label, 'r') as f:
            label_dict = json.load(f)
        k = 0
        for mess in label_dict['chars']:
            transcription = mess['transcription']
            print("label:{}|length:{}|step:{}|i:{}|transcription:{}".format(label, length, j, i, transcription))
            if dictionary.__contains__(transcription) == False:
                dictionary[transcription] = i
                i += 1
            coordinate = []
            coordinate.extend(mess['points'][0:2])
            coordinate.extend(mess['points'][4:6])
            print('coordinate:{}'.format(coordinate))
            num = dictionary[transcription]
            if isinstance(num, int) == False:
                print('num:{}'.format(num))
                pdb.set_trace()
            print('num:{}|k:{}'.format(num, k))
            img.crop(coordinate).save(save_path + label.split('.')[0] + '.' + str(k) + '.' + str(num) + '.jpg')
            k += 1
        print('one step finished')
    dictionary_inv = {v: k for k, v in dictionary.items()}
    with open(path + 'dictionary_inv.json', 'w') as f:
        json.dump(dictionary_inv, f)
    with open(path + 'dictionary.json', 'w') as f:
        json.dump(dictionary, f)
        print('end the length of the dictionary:{}'.format(len(dictionary)))  # 4135


if __name__ == '__main__':
    crop_img()
