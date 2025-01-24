from SC_VCG_multiprocess import *
from random import shuffle
from coptpy import *
import multiprocessing as mp
import matplotlib.pyplot as plt
import random,copy,time,pickle
import numpy as np


def list_shuffle(bids,escrow_prices,platform_bids,n,agent_num):
    #列表随机分组
    list = [i for i in range(agent_num)]
    shuffle(list)           #重排序
    bids_new = [[]for i in range(n)]
    escrow_prices_new = [[]for i in range(n)]
    platform_bids_new = [[]for i in range(n)]
    m = int(len(list)/n)
    list_shuffle = []
    for i in range(0, len(list), m):
        list_shuffle.append(list[i:i+m])
    i=0
    for singel_list in list_shuffle:
        for index in singel_list:
            bids_new[i].append(bids[index])
            escrow_prices_new[i].append(escrow_prices[index])
            platform_bids_new[i].append(platform_bids[index])
        i += 1
    return bids_new,escrow_prices_new,platform_bids_new,m


def batches_compute(input_batches):
    bids_new,escrow_prices_new,platform_bids_new,n,agent_num,total_t,m = input_batches
    #分批拍卖
    record_batches = {}
    x = split_integer(total_t,n)                        #将整数t拆分为3组
    #针对每个批次分别进行拍卖并存储结果
    record_t,W_N,profits,payment,utility,real_exchange,resX,resY,agents_profits = {},{},{},{},{},{},{},{},{}
    for i in range(n):
        record_t[i],W_N[i],profits[i],payment[i],utility[i],real_exchange[i],resX[i],resY[i] = SC_VCG_escrow(m,bids_new[i],\
                                                    platform_bids_new[i],escrow_prices_new[i],x[i])
        agents_profits[i] = W_N[i] - profits[i]
    total_W_N = sum(W_N.values())                                #社会福利social welfare
    total_platform_surplus = sum(profits.values())               #平台收益platform surplus
    total_agents_profits = sum(agents_profits.values())          #代理人收益 agents profits
    total_RE = sum([real_exchange[i]/m for i in range(n)])/n     #实际交换数量占总体agent数量的百分比
    total_PE = sum([len(resY[i])/m for i in range(n)])/n         #平台托管的数量占总agent数量的百分比
    record_batches[total_t] = [total_W_N,total_platform_surplus,total_agents_profits,total_RE,total_PE]
    return record_batches

def SCE_VCG_compute(input_scevcg):
    agent_num,bids,platform_bids,escrow_prices,total_t = input_scevcg
    #整批拍卖
    record_SCE = {}
    record_t,W_N,profits,payment,utility,real_exchange,resX,resY,agents_profits = {},{},{},{},{},{},{},{},{}
    record_t,W_N,PS,payment,utility,real_exchange,resX,resY = SC_VCG_escrow(agent_num,bids,platform_bids,escrow_prices,total_t)
    agent_profits = W_N - PS
    RE = real_exchange/agent_num
    PE = len(resY)/agent_num
    record_SCE[total_t] = [W_N,PS,agent_profits,RE,PE]
    return record_SCE

def multiprocess_compute(agent_num,bid_num,self_bid_range,other_bid_range,platform_bid_range,n,cycle,min_t):
    before = time.perf_counter()
    record_batches,record_SCE = [],[]
    for c in range(0,cycle):
        batches,SCE_VCG = {},{}
        #每一轮次生成新的投标价值、托管成本、平台估价
        bids,escrow_prices,platform_bids = gen_all_value(agent_num,bid_num,self_bid_range,other_bid_range,platform_bid_range,1)  
        #整批计算
        input_SCE = [(agent_num,bids,platform_bids,escrow_prices,t)for t in reversed(range(min_t,agent_num+1))]   #规模控制数t的计算界限
        
        pool = mp.Pool(8)
        SCE_VCG_dict = pool.map(SCE_VCG_compute,input_SCE)           #这个对应的函数没有使用
        pool.close()
        pool.join()
        
        for r in SCE_VCG_dict:
            SCE_VCG.update(r)
        record_SCE.append(SCE_VCG)
        
        #分批计算
        bids_new,escrow_prices_new,platform_bids_new,m = list_shuffle(bids,escrow_prices,platform_bids,n,agent_num)     #随机打乱列表
        input_batches = [(bids_new,escrow_prices_new,platform_bids_new,n,agent_num,t,m) for t in reversed(range(min_t,agent_num+1))]#规模控制数t的计算界限
        pool = mp.Pool(8)
        batches_SCE_VCG = pool.map(batches_compute,input_batches)
        pool.close()
        pool.join()
        
        for r in batches_SCE_VCG:
            batches.update(r)                 #将进程池返回的列表重新表达为字典
        record_batches.append(batches)
    after = time.perf_counter()
    print(f'total time:{after - before}')
    return record_batches,record_SCE

def save_pkl(name,agent_num,bid_num,dic,n):
    path = './data/'
    file_name = path + str(agent_num) + '_' + str(bid_num) + '_' + str(n)+'_'+name + '.pkl'
    # 字典保存
    f_save = open(file_name, 'wb')
    pickle.dump(dic, f_save)
    f_save.close()
    
#计算结果的读取
def read_pkl(name,agent_num,bid_num,n):
    path = './data/'
    file_name = path + str(agent_num) + '_' + str(bid_num) +'_'+ str(n)+'_'+ name + '.pkl'
    # 读取
    f_read = open(file_name, 'rb')
    dict_r = pickle.load(f_read)
    f_read.close()
    return dict_r