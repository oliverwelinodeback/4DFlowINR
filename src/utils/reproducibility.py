import torch
import numpy as np
import random
import os
import shutil


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def copy_source_code(model_dir, directory_to_backup=["."], folder_name="backup_source"):

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    print("Copying source code to model directory...")

    # Copy all the source file to the model dir for backup
    for directory in directory_to_backup:
        files = os.listdir(directory)
        for fname in files:
            if fname.endswith(".py"):
                dest_fpath = os.path.join(model_dir, folder_name, directory, fname)
                os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                shutil.copy2(f"{directory}/{fname}", dest_fpath)

    return 
