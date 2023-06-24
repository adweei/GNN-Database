import json
from os import listdir
from datetime import datetime

user_prof = dict()
files = listdir('ElonMusk/2023-02-08/user_profile/')
files = [file for file in files if '.json' in file]
now = datetime.now()
print(now)
for file in files:
    with open('ElonMusk/2023-02-08/user_profile/' + file, 'r') as fp:
        x = json.load(fp= fp)
        print(file, '||', len(x.keys()))
        user_prof.update(x)

print('Total progres :', len(user_prof.keys()))