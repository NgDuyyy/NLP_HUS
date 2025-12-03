import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

base = os.path.dirname(__file__)
data_dir = os.path.join(base, 'hwu')

files = {
    'train': os.path.join(data_dir, 'train.csv'),
    'val': os.path.join(data_dir, 'val.csv'),
    'test': os.path.join(data_dir, 'test.csv'),
}

for k,v in files.items():
    if not os.path.exists(v):
        raise FileNotFoundError(f"Expected file not found: {v}")

# Read CSVs (assume tab-separated or comma-separated - detect by extension/content)
# Inspect first line to decide separator
def detect_sep(path):
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline()
    if '\t' in first:
        return '\t'
    # default to comma
    return ','

sep = detect_sep(files['train'])
print('Detected separator for HWU files:', repr(sep))

# The HWU files appear to be two columns: text \t intent (or comma)
colnames = ['text', 'intent']

df_train = pd.read_csv(files['train'], sep=sep, header=None, names=colnames, encoding='utf-8')
df_val = pd.read_csv(files['val'], sep=sep, header=None, names=colnames, encoding='utf-8')
df_test = pd.read_csv(files['test'], sep=sep, header=None, names=colnames, encoding='utf-8')

print('Train shape:', df_train.shape)
print('Validation shape:', df_val.shape)
print('Test shape:', df_test.shape)

# Show some examples
print('\nTrain head:')
print(df_train.head())

# Label encoding across all splits
le = LabelEncoder()
all_intents = pd.concat([df_train['intent'], df_val['intent'], df_test['intent']]).astype(str)
le.fit(all_intents)

# Transform
for name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
    df['intent_id'] = le.transform(df['intent'].astype(str))

# Print label mapping and counts
mapping = {int(i): label for i,label in enumerate(le.classes_)}
print('\nNumber of classes:', len(mapping))
print('Label classes (index -> label) sample:')
# print first 20
for idx, lbl in list(mapping.items())[:20]:
    print(idx, lbl)

# Save mapping
with open(os.path.join(base, 'hwu_label_mapping.json'), 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print('\nSaved label mapping to hwu_label_mapping.json')

# Optionally save encoded csvs
out_dir = os.path.join(base, 'hwu_encoded')
os.makedirs(out_dir, exist_ok=True)

df_train.to_csv(os.path.join(out_dir, 'train_encoded.csv'), index=False)
df_val.to_csv(os.path.join(out_dir, 'val_encoded.csv'), index=False)
df_test.to_csv(os.path.join(out_dir, 'test_encoded.csv'), index=False)
print('Saved encoded CSVs to', out_dir)
