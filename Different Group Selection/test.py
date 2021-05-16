import numpy as np
import pickle


vali_df = pickle.load(open('./yelp-4/valiing_df.pkl','rb'),encoding="latin1")  # for validation
ids = pickle.load(open('./yelp-4/item_idd_list.pkl','rb'),encoding="latin1")  # for validation
#item_idd_genre_list = pickle.load(open('./ml1m-6/item_idd_genre_list.pkl','rb'),encoding="latin1")

item_idd_genre_list = pickle.load(open('./Yelp-4-randomgroup.pkl','rb'))
key_genre = ["g1","g2","g3","g4"]
nitems = [0]*4
nfeedback = [0]*4
for j in item_idd_genre_list:
	for g in j:
		nitems[key_genre.index(g)] = nitems[key_genre.index(g)]+1

for item_id in vali_df['item_id']:
	for g in item_idd_genre_list[item_id]:
		nfeedback[key_genre.index(g)] = nfeedback[key_genre.index(g)]+1
	

fdi = [nfeedback[i]/nitems[i] for i in range(4)]

data = None
with open('Rec_Yelp-4_BPR.mat','rb') as f:
	data = np.load(f)

recomends = []
user_num = data.shape[0]
item_num = data.shape[1]
for u in range(user_num):  # iterate each user
	u_pred = data[u, :]
	top15_idx = np.argpartition(u_pred, -15)[-15:]
	for t in top15_idx:
		recomends.append([u,t])
print("ok")

item_count_rec = [0]*item_num
item_count_vali = [0]*item_num
for rec in recomends:
	item_count_rec[rec[1]] = item_count_rec[rec[1]]+1
for it in vali_df['item_id']:
	item_count_vali[it] = item_count_vali[it]+1

print("ok")
#G = ["Sci-Fi","Adventure","Crime","Romance","Children's","Horror"]
G = key_genre
for genre in G:
	
	bias = []
	bias_ind = []
	have = []
	should = [] 
	for it in range(item_num):
		if genre not in item_idd_genre_list[it]:
			continue
		bias.append(item_count_vali[it]-item_count_rec[it])
		have += [item_count_rec[it]]
		should += [item_count_vali[it]]
		bias_ind.append(it)
		
	best_ind = bias.index(max(bias))
	best = bias_ind[best_ind]
	print("Most under/over recommended item in %s is %s (rec=%d,vali=%d)"%(genre,ids[best],have[best_ind],should[best_ind]))