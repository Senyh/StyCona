import pandas as pd
import random
import os
import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

seed_torch(seed=1234)

data_path = 'YOUR_DATA_PATH'
csv_files = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.endswith('csv')]
for i in range(len(csv_files)):
    input_csv = csv_files[i]
    save_list = input_csv.replace('csv', 'list')
    df = pd.read_csv(input_csv)
    data_list = df.values.tolist()
    if input_csv.find('train') != -1:
        random.shuffle(data_list)
    filename = open(save_list, 'w')
    filename.writelines([','.join(sample) + '\n' for sample in data_list])
    filename.close()

    with open(save_list, "r") as f1:
        patient_list = f1.readlines()
    patient_list = [item.replace("\n", "") for item in patient_list]
    print(patient_list[0].split(',')[0])