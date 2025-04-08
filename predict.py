import argparse
from pathlib import Path
import os

from dataframe import CustomDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

SUBMISSION_FILE_NAME = 'submission.csv'
SUBMISSION_DIR = 'submissions'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
parser.add_argument('-n', '--model_name', help='set a model name (optional parameter)')
parser.add_argument('-h', '--head_name', help='set a classification head name (optional parameter)')
args = parser.parse_args()

model_path = Path(args.model_path)
assert(model_path.excists())

model_name = args.model_name
if model_name is not None and model_name != "":
    classification_head_name = args.head_name
    if classification_head_name is None or classification_head_name = "":
        classification_head_name = 'default'
    
    submission_path = Path(SUBMISSION_DIR)
    if not submission_path.excists():
        os.mkdir(submission_path.absolute())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_csv_file = './data/test.csv'
test_images_dir = './data/images'

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # специальные значение нормализации для resnet
])

test_dataset = CustomDataset(test_csv_file, 
                             test_images_dir, 
                             classes_list, 
                             test_transform, # should be commented to print image
                             is_test=True)

# Определяем загрузчик данных
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Загрузка весов модели
model.load_state_dict(torch.load(model_path.absolute()))
model.to(device)

model.eval()
test_predictions = []

with torch.no_grad():
    for inputs in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs).cpu().numpy()
        
        test_predictions.append(preds)
# Concatenate all predictions
test_predictions = np.concatenate(test_predictions, axis=0)

# Create submission dataframe
submission_df = pd.read_csv(test_csv_file)
for i, class_name in enumerate(classes_list):
    submission_df[class_name] = test_predictions[:, i]

# Save submission file
if model_name:
    save_file_name = f"{submission_path.absolute()}/{model_name}_{classification_head_name}_submission.csv"
    submission_df.to_csv(save_file_name, index=False)
else:
    submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")