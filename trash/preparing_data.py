import json
import os

path_image = "./train_images"
train_gts = "./train_gts"
file_name = []
temp_arr = []
with open("instances_default.json", 'r') as f:
    data = json.load(f)
    
# for key, value in data.items():
#     print(key)
def make_gts(temp_arr):
    """Output: file_list là danh sách tất cả các file trong path và trong tất cả các
       thư mục con bên trong nó. dir_list là danh sách tất cả các thư mục con
       của nó. Các output đều chứa đường dẫn đầy đủ."""

    train_gts = "./train_gts"
    if not os.path.exists(train_gts):
        os.makedirs(train_gts)
    for temp in temp_arr:
        file_name, box = temp
        path_file_gts = train_gts+'/'+file_name+(".txt")
        
        mainline = f"{box[0]},{box[1]},{box[2]},{box[3]}"
        # for element in data:
        #     box = element['box']
        #     box_str = f'{box[0]},{box[1]},{box[2]},{box[1]},{box[2]},{box[3]},{box[0]},{box[3]}'
        #     #box_str = ",".join(map(str, box))
        #     text = element['text']
        #     line = box_str + ","+text + "\n"
        #     mainline += line
        with open(path_file_gts, mode='w') as f:
            f.write(mainline)
    # print("hello")


for element in data["images"]:
    id = element['id']
    file_name.insert(id,element['file_name'])
    #print(element['file_name'])
    #print(element)

for element in data["annotations"]:
    # id = element['id']
    # temp = [file_name[id],element['bbox']]
    # print(temp)
    # temp_arr.append(temp)
    try:
        id = element['id']
        temp = [file_name[image_id],element['segmentation']]
        # print(temp)
        temp_arr.append(temp)
    except:
        print("Can't fine file by id ",id)
    # print(element['bbox'])
#print(temp_arr)

make_gts(temp_arr)
