import os
def load_file(ROOT_PATH):
    list_path = []
    for i in os.listdir('./'+ROOT_PATH):
        list_path.append(ROOT_PATH+'/'+i)
    return list_path