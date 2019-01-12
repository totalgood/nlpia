# display an image/heatmap as 8bit integers in a grid
from PIL import Image
import numpy as np
import pandas as pd
import sys


# filepath = 'telephone_pole.jpg'

# im = Image.open(filepath)
# im.show()

# mat = np.array(list(im.getdata()))


def rescale(a, max_value=128, min_value=0, dtype='uint8', invert=False):
    typ = float if 'float' in str(a.dtype) else int
    print(typ)
    invert = -1 if invert else 1
    a = a.astype(float)
    a = invert * np.array(a)
    a -= np.min(a)
    a *= max_value / np.max(a)
    a += min_value
    return a.astype(dtype)





if __name__ == '__main__':
    defaults = 'ch07_telephone_pole.jpg', '128', '0', 'uint8'
    args = sys.argv[1:]
    for (i, (arg, default_value)) in enumerate(zip(args, defaults)):
        args[i] = type(default_value)(arg)
    for default_value in defaults[len(args):None]:
        args.append(default_value)
    filepath, max_value, min_value, dtype = args
    max_value, min_value = int(args[1]), int(args[2])
    args = filepath, max_value, min_value, dtype
    im = Image.open(args[0])
    im = im.convert('L')  # see help(im.convert) for modes
    im.show()
    mat = np.array(list(im.getdata()))
    mat = mat.reshape((im.height, im.width))
    # im = Image.fromarray(mat, mode='L')   # 'L' (Grayscale), 'RGB' and 'CMYK'
    # im.show() 
    print(args)
    mat = rescale(mat, *args[1:])
    mat_str = telephone_pole_nums = str(pd.DataFrame(mat))
    print(mat_str)
    with open(filepath[:-4] + '.txt', 'wt') as fout:
        fout.write(mat_str)
