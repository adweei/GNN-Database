# #test remove
# #將可能超過目標數的tweet轉推紀錄的檔名紀錄
# # np.load('D:\\GCN_Twitter\\ElonMusk\\2023-02-16\\retweet_distribution\\0_graph_retweet_distribution_rewrite.npy')
# raw_retweet_record_file = []                                                                                
# for path in os.listdir(test_tweet_retweet_distribution):
#     if os.path.isfile(os.path.join(test_tweet_retweet_distribution, path)):
#             raw_retweet_record_file.append(path)

# for nubmer in range(len(raw_retweet_record_file)):                                                          
#     feature = np.load(test_tweet_retweet_distribution + str(nubmer) + '_graph_retweet_distribution_rewrite.npy')
#     # print('號碼 {} 的graph 有 {} 的轉推數'.format(nubmer, len(np.where(feature[:,9] == 1)[0])))
#     #計算有哪些tweet有超過標準，這邊標準為" 1200人 "
#     if len(np.where(feature[:,9] == 1)[0]) < 1000 :
#         # print('號碼 {} 的graph 有 {} 的轉推數'.format(nubmer, len(np.where(feature[:,9] == 1)[0])))
#         os.remove(test_tweet_retweet_distribution + str(nubmer) + '_graph_retweet_distribution_rewrite.npy')
#     print('號碼 {} 的graph 有 {} 的轉推數'.format(nubmer, len(np.where(feature[:,9] == 1)[0])))



# #將2/16~20的label給建立出來

# #把要篩選的tweetid給讀入
# target_tweet = pd.read_excel(elon_mask_test_graph_data_dir + 'compare4.xlsx')   
# #retweet >= 1500 的推文數 0~28是29篇 
# thershold = 28                                                                                              
# target_tweet = target_tweet.drop(index = list(range(thershold + 1, len(target_tweet))))
# target_tweet['id'] = target_tweet['id'].astype(str)
# #user id讀入
# id = open(elon_mask_base_graph_data_dir + 'Graph/encoding_table.json')           
# user_id = json.load(id)
# id.close()
# #取兩個set之間的交集，藉以找出重合的target user(哪些人有追蹤這篇推文)
# user_id_list = list(user_id.keys())                                                                           
# user_id_list = [int(x) for x in user_id_list]
# user_id_set = set(user_id_list)
# print(len(user_id_list))

# # folder path                                                                                          #將各推文轉推情形的檔名紀錄
# tweet_retweet_path = elon_mask_test_graph_data_dir + 'retweeters/'
# # list to store files
# all_tweet_retweet_file = []
# # Iterate directory
# for path in os.listdir(tweet_retweet_path):
#     # check if current path is a file
#     if os.path.isfile(os.path.join(tweet_retweet_path, path)):
#         all_tweet_retweet_file.append(path)

# all_label = list()
# label_order = list()
# for file in all_tweet_retweet_file:                                                                     #在所有的json檔中做尋找
#     retweet_file = open(tweet_retweet_path + file)
#     part_tweet_retweet_file = json.load(retweet_file)
#     for per_tweet in part_tweet_retweet_file.keys():
#         inner = target_tweet['id'].isin([per_tweet])                                                    #檢查當前tweet是否在目標tweet名單內
#         if (inner.sum() != 0):                                                                          #有的話紀錄它在原本檔案中的index和有哪些人轉推
#             label_order.append(target_tweet[target_tweet['id'].isin([per_tweet])].index.tolist())
#             target_tweet_retweet = user_id_set.intersection(set(part_tweet_retweet_file[per_tweet]))
#             all_label.append(target_tweet_retweet)

# for number in range(0,len(all_label),1):
#     per_tweet_label = np.zeros(len(user_id))
#     for user in all_label[number]:
#         per_tweet_label[user_id[str(user)]] = 1
#     np.save(elon_mask_test_graph_data_dir + 'label/label_for_' + str(label_order[number][0]), per_tweet_label)


