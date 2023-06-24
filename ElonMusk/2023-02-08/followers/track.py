import json
from os import listdir

files = listdir('ElonMusk/2023-02-08/followers')
files = [i for i in files if '.json' in i]
total = dict()
for f in files:
    with open('ElonMusk/2023-02-08/followers/' + f, 'r') as fp:
        obj = json.load(fp= fp)
        print(f, '||', len(obj))
        total.update(obj)
print("Total ||", len(total))