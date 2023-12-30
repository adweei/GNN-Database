import sys
sys.path.append("D:/GCN_Twitter/ElonMusk/2023-02-16")
sys.path.append("D:/GCN_Twitter/ElonMusk/2023-02-20")
import pandas as pd

elon_mask_test_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-20/'
elon_mask_base_graph_data_dir = 'D:/GCN_Twitter/ElonMusk/2023-02-16/'

path = elon_mask_base_graph_data_dir + 'unity.xlsx'
outpath = 'D:/GCN_Twitter/ElonMusk/2023-02-20/data_graph/base_graph_tag.txt'

label_count = 27

# 自動產咒語
unity = pd.read_excel(path)     #pd.read_excel(elon_mask_base_graph_data_dir + 'unity.xlsx') 
tweet_id_list = list()
post_tweet_id_list = list()
reply_someone_tweet_id_list = list()
reply_own_tweet_id_list = list()
post_tweet_count_list = list()
reply_someone_tweet_count_list = list()
reply_own_tweet_count_list = list()
code_name = list()
label_num_list = list()
# 將tweet id讀入
for i in range(label_count):
    label_list = list()
    multilabel = 0
    if unity['post_own_tweet'][i] == 1:
        post_tweet_id_list.append(i)
        post_tweet_count_list.append(unity['retweet_count'][i])
        code_name.append(1)
    elif unity['reply_someone_tweet'][i] == 1:
        reply_someone_tweet_id_list.append(i)
        reply_someone_tweet_count_list.append(unity['retweet_count'][i])
        code_name.append(2)
    else:
        reply_own_tweet_id_list.append(i)
        reply_own_tweet_count_list.append(unity['retweet_count'][i])
        code_name.append(3)
    for j in range(4, 11, 1):
        label_list.append(unity.loc[i][j])

    for lable_num in range(len(label_list)): 
        if sum(label_list) == 1:     
            if label_list[lable_num] == 1:
                label_num_list.append(lable_num + 1)
        else:
            if label_list[lable_num] == 1:
                multilabel = multilabel * 10 + (lable_num + 1)
    # print(multilabel)
    if multilabel != 0:
        label_num_list.append(multilabel)

    print("label list: " + str(i), label_list)
print('label num list: ', label_num_list)


# print('post_own_tweet: ', post_tweet_id_list)
# print('post_own_tweet_count: ', post_tweet_count_list)
# print('reply_someone_tweet: ', reply_someone_tweet_id_list)
# print('reply_someone_tweet_count: ', reply_someone_tweet_count_list)
# print('reply_own_tweet: ', reply_own_tweet_id_list)
# print('reply_own_tweet_count: ', reply_own_tweet_count_list)
# print('code name: ', code_name)

with open(outpath, 'w') as f:
    for i in range(label_count):
        f.write(str(i) + ": " + str(code_name[i]) + " " + str(label_num_list[i]) + "\n")
f.close