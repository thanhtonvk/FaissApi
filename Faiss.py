from unittest import result
from Preprocess import img_preprocess,norm_mean_std
import cv2
import faiss
from LoadFile import load_file
import sys
from scipy import ndimage

def search(ROOT_PATH,pred_model):
    list_path = load_file(ROOT_PATH)
    list_result = []
    for idx,path in enumerate(list_path):
        list_img= []
        list_path_tmp = []
        original = cv2.imread(path)
        for idx2,path2 in enumerate(list_path):
            if(idx==idx2): continue
            # xoay ảnh
            img = cv2.imread(path2)
            for i in range(0,360,30):
                #rotation angle in degree
                rotated = ndimage.rotate(img, i)
                list_img.append(rotated)
                list_path_tmp.append(path2)

        flower_index = faiss.IndexFlatL2(128)
        fea_indexes = []
        for img_index, img_fp in enumerate(list_img):
            img = img_preprocess(img_fp, expand=True)
            embedded = pred_model.predict(img,verbose=False)  
            flower_index.add(embedded)
            fea_indexes.append(img_index)

        img_prep = img_preprocess(original, expand=True)
        test_fea = pred_model.predict(img_prep) 
        f_dists, f_ids = flower_index.search(test_fea, k=1)
        result = {"template":path,"dist":f_dists[0][0],"predict":list_path_tmp[f_ids[0][0]]}
        list_result.append(result)
    idx_result = 0
    dist_min = sys.float_info.max
    for idx,result in enumerate(list_result):
        if result['dist']<dist_min:
            dist_min = result['dist']
            idx_result = idx
    return list_result[idx_result]