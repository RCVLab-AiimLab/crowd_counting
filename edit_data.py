import scipy.io
import os
import copy
path = '/content/drive/My Drive/Queens/Crowdcounting_CARLA/dataset'
in_path1 = os.listdir(path)
for path1 in in_path1:
    in_path2 = os.listdir(os.path.join(path,path1))
    for path2 in in_path2:
        file_names = os.listdir(os.path.join(path,path1,path2,'ground_truth_wrong'))
        for file_name in file_names:
            if file_name.endswith(".mat"):
                mat = scipy.io.loadmat(os.path.join(path,path1,path2,'ground_truth_wrong',file_name))
                mat2 = copy.deepcopy(mat)
            # print(mat2['image_info'][0][0][0][0][0][0][1])
                for i in range(len(mat['image_info'][0][0][0][0][0])):
                    mat2['image_info'][0][0][0][0][0][i][1] = mat['image_info'][0][0][0][0][0][i][0]
                    mat2['image_info'][0][0][0][0][0][i][0] = mat['image_info'][0][0][0][0][0][i][1]
                # print(len(mat['image_info'][0][0][0][0][0]))
                # print(mat['image_info'][0][0][0][0][0][4][0])
                if not os.path.exists(os.path.join(path,path1,path2,'ground_truth')):
                    os.makedirs(os.path.join(path,path1,path2,'ground_truth'))
                scipy.io.savemat(os.path.join(path,path1,path2,'ground_truth',file_name), mat2)
                print(os.path.join(path,path1,path2,'ground_truth',file_name))

      