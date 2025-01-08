import torch
import  os
import time
from datetime import datetime

timestamp = time.time()
cur_time = datetime.now()
cur_path = os.path.abspath(os.path.curdir)
# model_path = os.path.join(cur_path, 'results', 'weights', 'googlenet')
# google_model = os.path.join(model_path, 'GoogleNet.pth')

# loaded_model = torch.load(google_model)
#
# print(loaded_model)


def save_model_info(model_folder):
    model_path = os.path.join(cur_path, model_folder)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return os.path.abspath(model_path)


def save_model_info_txt():
    cur_path = os.path.abspath(os.path.curdir)
    for net_name in os.listdir(os.path.join(cur_path, 'results', 'weights')):
        model_path = os.path.abspath(os.path.join(cur_path, 'results', 'weights', net_name))
        print(model_path)
        model_file = os.listdir(model_path)  # ['AlxNet.pth']\['GoogleNet.pth']
        for model_pth in model_file:
            print(model_pth)
            split_file = os.path.splitext(model_pth)[0]  # 以.分割成两个字符串，‘AlxNet‘ ’.pth‘
            loaded_pth = torch.load(os.path.join(model_path, model_pth))

            state_dict_str = str(loaded_pth)
            save_model_info('model_info')
            with open(os.path.join(save_model_info('model_info'), f'{split_file}_model_info + {timestamp}.txt'), 'w') as file:
                file.write(state_dict_str)
                # # 如果选择覆盖，继续写入；如果选择不覆盖，则停止
                # if input("是否覆盖(y/n)?: ").lower() == 'y':
                #     with open(os.path.join(cur_path, 'model_info', f'{split_file}_model_info.txt'), 'a') as file_a:
                #         file_a.write(state_dict_str)
                # else:
                #     print("操作已取消，未做更改.")
                # file_a.close()
            file.close()


save_model_info_txt()
