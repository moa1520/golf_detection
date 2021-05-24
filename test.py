import os
from glob import glob
import json

root = '/media/tk/SSD_250G'
images = sorted(os.listdir(os.path.join(root, "image")))
labels = sorted(os.listdir(os.path.join(root, "label/bb")))

label_file = os.path.join(root, 'label/bb', labels[0])

f = open(label_file)
label = json.loads(f.read())
person_bb = label[label_file.split(
    '_')[-1].split('.')[0] + '_detect.json'][1]['bb']['person_bb']
club_bb = label['swing001_detect.json'][1]['bb']['club_bb']
ball_bb = label['swing001_detect.json'][1]['bb']['ball_bb']
print(person_bb)
