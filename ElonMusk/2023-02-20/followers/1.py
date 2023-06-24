import json

follow_list = dict()
edges = 0
node = 0
with open('ElonMusk/2023-02-20/target_users.json', 'r') as fp:
    target_user = json.load(fp= fp)

file_path = 'ElonMusk/2023-02-20/followers/'
for i in range(14):
    with open(file_path + '%d_followers.json' % i, 'r') as fp:
        follow_list.update(json.load(fp= fp))

exist = list(follow_list.keys())
exist = [int(i) for i in exist]
diff = set(target_user).difference(set(exist))
node = len(exist)
for k in follow_list.keys():
    edges += len(follow_list[k])
#print(diff, node, edges)

print(follow_list['18856867'])