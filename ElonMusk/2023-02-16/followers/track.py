import json

s = 0
for i in range(14):
    with open('ElonMusk/2023-02-16/followers/%d_followers.json' % i, 'r') as fp:
        obj = json.load(fp= fp)
        for k in obj:
            s += len(obj[k])
print(s)