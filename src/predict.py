import argparse
from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import extract_model_name
from utils.constants import (models,
                             classes_list,
                             classification_heads,
                             test_loader,
                             test_csv_file)

SUBMISSION_FILE_NAME = 'submission.csv'
SUBMISSION_DIR = 'submissions'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
args = parser.parse_args()

model_path = Path(args.model_path)
assert(model_path.exists())
    
model_file_path = Path(args.model_path)
assert(model_file_path.exists())

model_name = extract_model_name(model_file_path.name)
assert(model_name is not None)
assert(model_name in models.keys())

# check if we use custom classification head
if 'head' in model_file_path.parent.name:
    head_name = model_file_path.parent.name
else:
    head_name = None
    
submission_path = Path(SUBMISSION_DIR)
if not submission_path.exists():
    os.mkdir(submission_path.absolute())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if head_name is None:
    model = models[model_name](n_classes=len(classes_list))
else:
    model = models[model_name](n_classes=len(classes_list), classification_head=classification_heads[head_name])
assert(model is not None)

# Загрузка весов модели
model.load_state_dict(torch.load(model_path.absolute()))
model.to(device)

print(f"Make submission with model {model_name} and classification head {head_name}")
model.eval()

propabilities = []
logits = []

for images in tqdm(test_loader):
    images = images.to(device)
    with torch.no_grad():
        prediction_batch = model(images)
    
    logits_batch = prediction_batch.data.cpu().detach()
    logits.extend(logits_batch.tolist())
    propabilities_batch = F.softmax(logits_batch, dim=1)
    propabilities.extend(propabilities_batch.tolist()) # получили список списков

df = pd.DataFrame(propabilities, columns=classes_list)
df = df.round(4)

submit_df = pd.read_csv(test_csv_file)
submit_df = pd.concat([submit_df, df], axis=1)

save_file_name = f"{submission_path.absolute()}/{model_name}_{head_name}_submission.csv"
submit_df.to_csv(save_file_name, index=False)

print("Submission file created successfully!")