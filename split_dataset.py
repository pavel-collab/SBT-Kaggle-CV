import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import shutil
from tqdm import tqdm

def move_img_to_dir(images: list, src_img_dir_path: str, dst_img_dir_path: str):
    try:
        for image in tqdm(images):
            shutil.copyfile(f"{src_img_dir_path}/{image}.jpg", f"{dst_img_dir_path}/{image}.jpg")
    except Exception as ex:
        print(f"[err] caught exception {ex}")
        
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_file_path', help='set a path to datafile')
args = parser.parse_args()

data_file_path = Path(args.data_file_path)
assert(data_file_path.exists())

data_root_directory = data_file_path.parent

df = pd.read_csv(data_file_path.absolute())
df_train, df_validation = train_test_split(df, test_size=0.3, random_state=42)

# df_train = df_train.reset_index().drop(columns=['index'])
# df_validation = df_validation.reset_index().drop(columns=['index'])

assert(data_root_directory.is_dir())

os.mkdir(f"{data_root_directory}/train")
os.mkdir(f"{data_root_directory}/validation")
os.mkdir(f"{data_root_directory}/train/images")
os.mkdir(f"{data_root_directory}/validation/images")

src_img_dir = Path(f"{data_root_directory}/images")
assert(src_img_dir.exists())

train_images_path = Path(f"{data_root_directory}/train/images")
validation_image_path = Path(f"{data_root_directory}/validation/images")
assert(train_images_path.exists())
assert(validation_image_path.exists())

move_img_to_dir(df_train['image_id'].to_list(), 
                src_img_dir.absolute(), 
                train_images_path.absolute())
move_img_to_dir(df_validation['image_id'].to_list(), 
                src_img_dir.absolute(), 
                validation_image_path.absolute())

df_train.to_csv(f"{data_root_directory}/train/train.csv", index=False)
df_validation.to_csv(f"{data_root_directory}/validation/validation.csv", index=False)