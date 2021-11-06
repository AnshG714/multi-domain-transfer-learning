# a utility script to format the sports domain data. Need only be run when the sports dataset is first added.
SPORTS_DATA_PATH = './datasets/sports'

import pandas as pd
import os
import math

data = pd.read_csv(os.path.join(SPORTS_DATA_PATH, "class_dict.csv"), usecols=["class_index", "class"], delimiter=',')
for i in range(len(data["class_index"])):
  if math.isnan(data["class_index"][i]):
    data = data.drop(i)

data.class_index = data.class_index.astype(int)
data = data.set_index("class")

train_df = pd.DataFrame(columns = ["img_name", "label"])
test_df = pd.DataFrame(columns = ["img_name", "label"])

train_path = os.path.join(SPORTS_DATA_PATH, 'train')
test_path = os.path.join(SPORTS_DATA_PATH, 'test')

sports = list(filter(lambda x: x[0] != '.', os.listdir(train_path)))

for sport in sports:
  images = os.listdir(os.path.join(train_path, sport))
  for i in range(10):
    train_df = train_df.append(pd.Series(data = {"img_name": sport + "/" + images[i], "label": data.loc[sport]["class_index"]}), ignore_index=True)

for sport in sports:
  images = os.listdir(os.path.join(test_path, sport))
  for i in range(5):
    test_df = test_df.append(pd.Series(data = {"img_name": sport + "/" + images[i], "label": data.loc[sport]["class_index"]}), ignore_index=True)


train_df.to_csv(os.path.join(SPORTS_DATA_PATH, 'train_csv.csv'))
test_df.to_csv(os.path.join(SPORTS_DATA_PATH, 'test_csv.csv'))