import json
from os import listdir, remove
s = 0
file_path = 'ElonMusk/2023-02-20/hashtags/timelines/'
files = listdir(file_path)
for f in files:
    with open(file_path + f, 'r') as fp:
        obj = json.load(fp= fp)
        k = list(obj.keys())[0]
        if not obj[k]:
            #remove(file_path + f)
            s += 1
            print(f)
print(s)
        