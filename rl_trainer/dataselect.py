import pickle
import os
file1="data"
file2="map"+str(4)
fileNamePath=os.getcwd()+"\\"+file1+"\\"+file2
data_set = {'obs': [], 'action': []}
data_temp = {'obs': [], 'action': []}
with open(fileNamePath+'\\' + 'data.pkl', 'rb') as f:
    data_temp=pickle.load(f)
'''
注意第n张png描述的是执行第n个aciton之后的观测，因此(o_{n-1},a_n)才是数据对，即Agent在观测o_{n-1}后执行a_n然后得到o_n，o_n上的黑线为a_n。
如果觉得o_n很好的话，说明(o_{n-1},a_n)对值得被加入总数据中，即：data_set中的第n-1对数据。
'''
with open('img/label.txt', 'r') as f:
    for line in f.readlines():
        if line == '\n':
            continue
        index=int(line.strip())
        data_set['obs'].append(data_temp['obs'][index-1])
        data_set['action'].append(data_temp['action'][index-1])

with open(fileNamePath + '\\' + 'all_data.pkl', 'wb') as f:
    pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

print(data_set['obs'][1])