##
# Depth Mapper
##




import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.preprocessing import normalize
import numpy as np
import sys

pic_path = sys.argv[1]
max_depth = int(sys.argv[2])

#get png pic and convert it to x and y diffs
pic_data = io.imread(pic_path)
try:
    pic_data = color.rgb2gray(color.rgba2rgb(pic_data))
except:
    pic_data = color.rgb2gray(pic_data)
pic_diff_x = -np.diff(pic_data)
pic_diff_y = -np.diff(pic_data, axis = 0)

#diff reduces the length of one dimension.
#add a row to each to make them the same shape.
pic_diff_xtrans = np.transpose(pic_diff_x)
pic_diff_xplus = np.transpose(
    np.concatenate(
        [
            pic_diff_xtrans, [np.zeros_like(pic_diff_xtrans[0])]
        ]
    )
)
pic_diff_yplus = np.concatenate([pic_diff_y, [np.zeros_like(pic_diff_y[0])]])

#zip up x and y diffs
pic_diff = max_depth*np.array(list(zip(pic_diff_xplus, pic_diff_yplus)))

#swap the axes to make it pic_diff_im[y,x,color] rather than [y,color,x]
pic_diff_im = np.swapaxes(pic_diff, 1,2)



#turn the pic_diff into a normal map
#shove all values together
pic_normals_before_normalizing = np.array(
    [
        [
            np.concatenate([pair,[1]])
            for pair in row
        ]
        for row in pic_diff_im
    ]
)

#divide by the abs of the vec3 length to get a normal
pic_normals = np.array(
    [
        [
            normalize([triplet])[0]
            for triplet in row
        ]
    for row in pic_normals_before_normalizing
    ]
)

#raise the values from [-0.5,0.5] to [0,1]
rgb_normals = 0.5*(pic_normals + 1) #or does this have to be 0.5*(pic_normals + 1)??

io.imsave(pic_path[:-4] + '_normals.png', rgb_normals)