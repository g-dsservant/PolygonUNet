import numpy as np
import os
import json
from PIL import Image

curr_path = os.getcwd()
inputs_path = os.path.join(curr_path, 'dataset', 'training', 'inputs')
outputs_path = os.path.join(curr_path, 'dataset', 'training', 'outputs')

#extracting all the colors in the dataset from outputs
colors = set()
for image in os.listdir(outputs_path):
    color = image.split('_')[0]
    colors.add(color)

encoded_color = {color:i for color, i in zip(list(colors), range(len(colors)))}

if __name__=='__main__':
	#saving a mapping between colors and their corresponding numerical representation as a dict
	filename = 'encoded_color_dict.json'
	with open(filename, 'w') as f:
	    json.dump(encoded_color, f, indent=4)

