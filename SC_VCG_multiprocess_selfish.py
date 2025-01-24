import random,copy,time,pickle
import numpy as np
import matplotlib.pyplot as plt
from coptpy import *
import multiprocessing as mp

#data---------------------------------------------------------------
#根据输入的min_price和max_price生成这范围内的num个数字
def gen_data(num,min_price,max_price,real_ratio):
    gen_value = []
    prices = []
    for i in range(num):
        price_flag = True
        while price_flag:
            price = random.randint(min_price*100, max_price*100)/100
            price *= real_ratio        #平台真实报价比例
            if price not in prices:    #生成与之前不一样的价格
                prices.append(price)
                gen_value.append(price)
                price_flag = False
    return gen_value

def gen_data_selfish(num,min_price,max_price,selfish_ratio):
    gen_value_true = []
    gen_value_selfish = []
    prices = []
    for i in range(num):
        price_flag = True
        while price_flag:
            price_true = random.randint(min_price*100, max_price*100)/100  #平台真实估价
            price_selfish = price_true*selfish_ratio                #平台提交报价（虚假）
            if price_true not in prices:    #生成与之前不一样的价格
                prices.append(price_true)
                gen_value_true.append(price_true)
                gen_value_selfish.append(price_selfish)
                price_flag = False

    return gen_value_selfish,gen_value_true

# 随机产生选择的agents序列|生成除agent_id本身的其他agent序列
def gen_objs(bid_num, agent_id,agent_num):
    output = []
    while len(output) < min(bid_num,agent_num-1):
        k = random.randint(0, agent_num-1)
        if k not in output and k != agent_id:
            output.append(k)
    return output

# 生成agents的bids,平台估值platform_bids，托管成本escrow_prices
def gen_all_value(agent_num,bid_num,self_bid_range,other_bid_range,platform_bid_range,platform_selfish_ratio):
    self_bid_value = gen_data(agent_num,self_bid_range[0],self_bid_range[1],real_ratio=1)
    other_bid_value = [[] for i in range(agent_num)]
    for i in range(agent_num):
        other_bid_value[i] = gen_data(agent_num,other_bid_range[0],other_bid_range[1],real_ratio=1)
        other_bid_value[i][i] = 0

    xor_bid = [[0 for i in range(agent_num)] for j in range(agent_num)]
    for i in range(agent_num):
        xor_bid[i][i] = 0.000001
        bid_id = gen_objs(bid_num, i,agent_num)
        for j in range(len(bid_id)):
            #投标价格 = 对别人的估价 - 对自己的估价
            xor_bid[i][bid_id[j]] = other_bid_value[i][bid_id[j]] - self_bid_value[i]

    escrow_prices = copy.deepcopy(self_bid_value)         #托管成本=对自己的商品估价
    platform_bids_selfish,platform_bids_true = gen_data_selfish(agent_num,platform_bid_range[0],platform_bid_range[1],platform_selfish_ratio)   #平台估价
    
    return xor_bid,escrow_prices,platform_bids_selfish,platform_bids_true

#model---------------------------------------------------------------
#SC+escrow的模型求解
def optimize(N,V,B,S,t):
    model = Envr().createModel()

    #设置变量
    x,y={},{}
    for i in range(N):
        y[i]=model.addVar(vtype="B")
        for j in range(N):
            x[i,j]=model.addVar(vtype="B")

    X = [[1 for j in range(N)]for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and V[i][j] == 0:X[i][j] = 0

    #设置目标
    model.setObjective(quicksum(x[i,j]*V[i][j] for i in range(N) for j in range(N))
                       +quicksum(y[i]*(B[i]-S[i])for i in range(N)),sense=COPT.MAXIMIZE)
    #添加约束
    for i in range(N):
        model.addConstr(quicksum(x[i,j] for j in range(N)) + y[i] == 1)
    for j in range(N):
        model.addConstr(quicksum(x[i,j] for i in range(N)) + y[j] == 1)

    model.addConstr(quicksum(x[i,i] for i in range(N)) >= N - t)

    for i in range(N):
        for j in range(N):
            model.addConstr(x[i,j] <= X[i][j])

    model.setParam(COPT.Param.TimeLimit, 150.0)
    model.setParam(COPT.Param.Logging,0)   #取消显示求解中间过程
    status = model.solve()
    # model.write("d:/1111111111111111.lp")
    W_N = model.objval
#     print(f"ObjValue={model.objval}")

    result_x,result_y=[],[]
    #result_x是选择交换的agent和对应的item编号，result_y是选择托管的agent序号列表
    if model.Status==COPT.OPTIMAL or model.Status in [COPT.TIMEOUT, COPT.NODELIMIT, COPT.INTERRUPTED]:
        for i in range(N):
            if y[i].value>=0.5:
                result_y.append(i)
            for j in range(N):
                if x[i,j].value>=0.5:
                    result_x.append((i,j))
#     print(f'result_x:{result_x},result_y:{result_y}')
    return W_N,result_x,result_y

#求去掉agent i的总社会福利(SC+escrow的模型)
def optimize_P(N,V,B,S,t,agent_id):
    model = Envr().createModel()

    #设置变量
    x,y={},{}
    for i in range(N):
        y[i]=model.addVar(vtype="B")
        for j in range(N):
            x[i,j]=model.addVar(vtype="B")

    X = [[1 for j in range(N)]for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and V[i][j] == 0:X[i][j] = 0

    #设置目标
    model.setObjective(quicksum(x[i,j]*V[i][j] for i in range(N) for j in range(N))
                       +quicksum(y[i]*(B[i]-S[i])for i in range(N)),sense=COPT.MAXIMIZE)
    #添加约束
    for i in range(N):
        model.addConstr(quicksum(x[i,j] for j in range(N)) + y[i] == 1)
    for j in range(N):
        model.addConstr(quicksum(x[i,j] for i in range(N)) + y[j] == 1)

    model.addConstr(quicksum(x[i,i] for i in range(N)) >= N - t)

    for i in range(N):
        for j in range(N):
            model.addConstr(x[i,j] <= X[i][j])
            
    model.addConstr(x[agent_id,agent_id] == 1)#限制agent i不能参与拍卖


    model.setParam(COPT.Param.TimeLimit, 150.0)
    model.setParam(COPT.Param.Logging,0)   #取消显示求解中间过程
    status = model.solve()

    W_N = model.objval
#     print(f"ObjValue={model.objval}")

    result_x,result_y=[],[]
    #result_x是选择交换的agent和对应的item编号，result_y是选择托管的agent序号列表
    if model.Status==COPT.OPTIMAL or model.Status in [COPT.TIMEOUT, COPT.NODELIMIT, COPT.INTERRUPTED]:
        for i in range(N):
            if y[i].value>=0.5:
                result_y.append(i)
            for j in range(N):
                if x[i,j].value>=0.5:
                    result_x.append((i,j))
#     print(f'result_x:{result_x},result_y:{result_y}')
    return W_N,result_x,result_y

#SC-VCG模型
def SC_VCG_optimize(N,V,t):   #agent_num,bids,scale control
    model = Envr().createModel()

    #设置变量
    x={}
    for i in range(N):
        for j in range(N):
            x[i,j]=model.addVar(vtype="B")
    X = [[1 for j in range(N)]for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and V[i][j] == 0:X[i][j] = 0

    #设置目标
    model.setObjective(quicksum(x[i,j]*V[i][j] for i in range(N) for j in range(N)), sense=COPT.MAXIMIZE)

    for i in range(N):
        model.addConstr(quicksum(x[i,j] for j in range(N)) == 1)
    for j in range(N):
        model.addConstr(quicksum(x[i,j] for i in range(N)) == 1)

    model.addConstr(quicksum(x[i,i] for i in range(N)) >= N-t)
    
    for i in range(N):
        for j in range(N):
            model.addConstr(x[i,j] <= X[i][j])

    model.setParam(COPT.Param.TimeLimit, 150.0)
    model.setParam(COPT.Param.Logging,0)   #取消显示求解中间过程
    status = model.solve()
    W_N_scvcg = model.objval
#     print(f"ObjValue={model.objval}")

    result=[]
    if model.Status==COPT.OPTIMAL or model.Status in [COPT.TIMEOUT, COPT.NODELIMIT, COPT.INTERRUPTED]:
        for i in range(N):
            for j in range(N):
                if x[i,j].value>=0.5:
                    result.append((i,j))
    return W_N_scvcg,result

#SC-VCG 求去掉agent i的总社会福利
def SC_VCG_optimize_P(N,V,t,agent_id):   #agent_num,bids,scale control
    model = Envr().createModel()

    #设置变量
    x={}
    for i in range(N):
        for j in range(N):
            x[i,j]=model.addVar(vtype="B")
    X = [[1 for j in range(N)]for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and V[i][j] == 0:X[i][j] = 0

    #设置目标
    model.setObjective(quicksum(x[i,j]*V[i][j] for i in range(N) for j in range(N)), sense=COPT.MAXIMIZE)

    for i in range(N):
        model.addConstr(quicksum(x[i,j] for j in range(N)) == 1)
    for j in range(N):
        model.addConstr(quicksum(x[i,j] for i in range(N)) == 1)

    model.addConstr(quicksum(x[i,i] for i in range(N)) >= N-t)

    for i in range(N):
        for j in range(N):
            model.addConstr(x[i,j] <= X[i][j])
        
    model.addConstr(x[agent_id,agent_id] == 1)#限制agent i不能参与拍卖

    model.setParam(COPT.Param.TimeLimit, 150.0)
    model.setParam(COPT.Param.Logging,0)   #取消显示求解中间过程
    status = model.solve()
    W_N_scvcg = model.objval
#     print(f"ObjValue={model.objval}")

    result=[]
    if model.Status==COPT.OPTIMAL or model.Status in [COPT.TIMEOUT, COPT.NODELIMIT, COPT.INTERRUPTED]:
        for i in range(N):
            for j in range(N):
                if x[i,j].value>=0.5:
                    result.append((i,j))
    return W_N_scvcg,result

#拍卖分配及定价过程-------------------------------------------------------------
#SC-VCG的拍卖分配和定价过程
def Scale_control_VCG(N,V,t):    
    W_N,result_scvcg = SC_VCG_optimize(N,V,t)
    #real_exchange,payment,utility,社会福利W_N
    real_exchange = 0
    fail_agent = []
    bid_value = 0
    for i in result_scvcg:
        if i[0] == i[1]:
            fail_agent.append(i[0])
        else:
            real_exchange += 1

    payment,utility = {},{}
    profits = 0
    for agent_id in range(N):
        if agent_id in fail_agent:continue
        for i in result_scvcg:
            if i[0] == agent_id:
                bid_value = V[agent_id][i[1]]    
        W_no_i,r_no_i = SC_VCG_optimize_P(N,V,t,agent_id) 
        utility[agent_id] = W_N - W_no_i   #agent profit
        payment[agent_id] = bid_value - utility[agent_id]     #platform surplus
        profits += payment[agent_id]

    print(f'Scale Control result:\nt:{t},W(N):{W_N}\nprofits:{profits}')
    return t,round(W_N,2),utility,payment,round(profits,2),real_exchange,result_scvcg

#SC-VCG+托管的拍卖分配和定价过程
def SC_VCG_escrow(agent_num,bids,platform_bids,platform_true_value,escrow_prices,t):
    W_N,result_X,result_Y = optimize(agent_num,bids,platform_bids,escrow_prices,t) #虚假报价对应的社会福利
    #real_exchange,payment,utility,社会福利W_N
    real_exchange = 0
    fail_agent = []
    for i in result_X:
        if i[0] == i[1]:
            fail_agent.append(i[0])
        else:
            real_exchange += 1
    real_exchange += len(result_Y)   #把物品交给平台托管也算一次交换

    payment,utility = {},{}       #定价；代理人收益
    profits = 0     #平台利润
    W_N_True = 0
    for agent_id in range(agent_num):
        if agent_id in fail_agent:continue
        if agent_id in result_Y:
            bid_value = -1*escrow_prices[agent_id]
            profits += platform_true_value[agent_id]
            W_N_True += (platform_true_value[agent_id] - escrow_prices[agent_id])
        else:
            for i in result_X:
                if i[0] == agent_id:
                    bid_value = bids[agent_id][i[1]]
                    W_N_True += bid_value
        W_no_i,r_X,r_Y = optimize_P(agent_num,bids,platform_bids,escrow_prices,t,agent_id)
        utility[agent_id] = W_N - W_no_i
        payment[agent_id] = bid_value-utility[agent_id]

        profits += payment[agent_id]

    print(f'Scale Control + escrow result: t:{t},W(N):{W_N_True},profits:{profits}')
    return t,round(W_N_True,2),round(profits,2),payment,utility,real_exchange,result_X,result_Y,W_N

# ----------------------------------------------------------------
#计算结果的保存
def save_pkl(name,agent_num,bid_num,prr,dic):
    path = './data/'
    file_name = path + str(agent_num) + '_' + str(bid_num) + '_'+ 'selfish_ratio'+str(prr) + '_' + name + '.pkl'
    # 字典保存
    f_save = open(file_name, 'wb')
    pickle.dump(dic, f_save)
    f_save.close()
    
#计算结果的读取
def read_pkl(name,agent_num,bid_num,prr):
    path = './data/'
    file_name = path + str(agent_num) + '_' + str(bid_num) + '_' + 'selfish_ratio'+str(prr) + '_'+ name + '.pkl'
    # 读取
    f_read = open(file_name, 'rb')
    dict_r = pickle.load(f_read)
    f_read.close()
    return dict_r

#-------------------------------------------------------------------
#单参数输入的计算函数，规模控制（为使用多进程计算）
def cycle_compute_SC_VCG(input_p):
    SC_VCG_s = {}
    agent_num,bids,t = input_p
    record_t,W_N,utility,payment,profits,real_exchange,result = Scale_control_VCG(agent_num,bids,t)
    agents_profits = W_N - sum(utility.values())
    SC_VCG_s[t] = [W_N,profits,payment,utility,real_exchange,result,agents_profits]
    return SC_VCG_s

#单参数输入的计算函数，规模控制+托管（为使用多进程计算）
def cycle_compute_SCE_VCG(input_p):
    SCE_VCG_s = {}
    agent_num,bids,platform_bids,platform_true_value,escrow_prices,t = input_p
    record_t,W_N,profits,payment,utility,real_exchange,resX,resY,W_N_selfish = SC_VCG_escrow(agent_num,bids,platform_bids,platform_true_value,escrow_prices,t)
    #
    agents_profits = W_N - sum(utility.values())     #utiliy用虚假的社会福利计算,W_N是真实的社会福利
    SCE_VCG_s[t] = [W_N,profits,payment,utility,real_exchange,resX,resY,agents_profits]
    return SCE_VCG_s


#主函数，负责生成数据并使用两种方式分配及定价，在重复cycle次后取均值，并存储数据
def compute(agent_num,bid_num,cycle,self_bid_range,other_bid_range,platform_bid_range,p_selfish_ratio):
    record_SCE_VCG = []
    max_t = agent_num
    before = time.perf_counter()
    for c in range(0,cycle):
        #每一轮次生成新的投标价值、托管成本、平台报价(selfish)、平台真实估价(true)（V,B,S)
        bids,escrow_prices,platform_bids,platform_true_value = gen_all_value(agent_num,bid_num,self_bid_range,other_bid_range,platform_bid_range,p_selfish_ratio)  
        # print(f'bids:{bids}\nplatform_bids:{platform_bids}\nescrow_prices:{escrow_prices}\n')
        
        #SC-VCG+escrow求解过程
        SCE_VCG = {}
        pool = mp.Pool(5)
        
        input_SCEVCG = [(agent_num,bids,platform_bids,platform_true_value,escrow_prices,t) for t in reversed(range(1,max_t+1))]
        SCE_VCG_r = pool.map(cycle_compute_SCE_VCG, input_SCEVCG)#并发计算形成列表
        for r in SCE_VCG_r:
            SCE_VCG.update(r)
        record_SCE_VCG.append(SCE_VCG)
  
    after = time.perf_counter()
    print(f'total time:{after - before}')

    mean_SCE_VCG = {}
    for c in range(0,cycle):
        W_N_E,profits_E,RE_E,AP_E,PE_E = 0,0,0,0,0
        for t in reversed(range(1,max_t+1)):
            W_N_E = record_SCE_VCG[c][t][0]                             #社会福利 social welfare
            profits_E = record_SCE_VCG[c][t][1]                         #平台盈余 platform surplus
            RE_E = record_SCE_VCG[c][t][4]/agent_num            #实际交换数量占总体agent数量的百分比
            AP_E = record_SCE_VCG[c][t][7]/agent_num            #agent's profits
            PE_E = len(record_SCE_VCG[c][t][6])/agent_num            #平台托管的数量占总agent数量的百分比
            print(f'SCE-VCG: t:{t},c:{c},W(N):{W_N_E},platform profits:{profits_E}')

    #计算两种方法的指标均值
    mean_SCE_VCG = {}
    for t in reversed(range(1,max_t+1)):
        W_N_E,profits_E,total_payment_E,RE_E,AP_E,PE_E = 0,0,0,0,0,0
        for c in range(0,cycle):
            W_N_E += record_SCE_VCG[c][t][0]                             #社会福利 social welfare
            profits_E += record_SCE_VCG[c][t][1]                         #平台盈余 platform surplus
            total_payment_E += sum(record_SCE_VCG[c][t][2].values())     #花费的总价 暂时不需要
            RE_E = RE_E + (record_SCE_VCG[c][t][4]/agent_num)            #实际交换数量占总体agent数量的百分比
            AP_E = AP_E + (record_SCE_VCG[c][t][7]/agent_num)            #agent's profits
            PE_E = PE_E + (len(record_SCE_VCG[c][t][6])/agent_num)            #平台托管的数量占总agent数量的百分比

        mean_SCE_VCG[t] = [W_N_E,profits_E,total_payment_E,RE_E,AP_E,PE_E]
        mean_SCE_VCG[t] = [x/cycle for x in mean_SCE_VCG[t]]
    #存储数据为pickle文件
    save_pkl('SCE-VCG',agent_num,bid_num,p_selfish_ratio,mean_SCE_VCG)

#对于Agent Prpfits的存储存在问题，后面可以用AP=Social Walfare-Platform Surplurs重新计算，但是有空还是要把代码改一次

# 拆分整数
def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n
