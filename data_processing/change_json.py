import os
import json
import numpy as np
import itertools


def my_linspace(l):
    new_x = []
    new_y = []
    for x in np.linspace(l[6], l[4], 7):
        new_x.append(x)
    for y in np.linspace(l[7], l[5], 7):
        new_y.append(y)
    for x in np.linspace(l[2], l[0], 7):
        new_x.append(x)
    for y in np.linspace(l[3], l[1], 7):
        new_y.append(y)
    print(new_x)
    print(len(new_x))
    print(new_y)
    print(len(new_y))
    new_l = list(itertools.chain.from_iterable(zip(new_x, new_y)))
    return new_l


filedir = r'./labels'
filenames = os.listdir(filedir)
for filename in filenames:
    filepath = filedir + '/' + filename
    print(filepath)

    after = {}
    with open(filepath, 'r') as f:
        data = json.load(f)
        mask = data["lines"]
        for m in mask:
            print(type(m))
            print(m["transcription"])
            print(m["points"])
            l = m["points"]
            new_l = my_linspace(l)
            print(new_l)
            print(len(new_l))
            m["points"] = new_l
        print(data)
        print(type(data))
        after = data
        f.close()

    with open(filepath, 'w') as f:
        json.dump(after, f)
        print("Success!")
        f.close()
