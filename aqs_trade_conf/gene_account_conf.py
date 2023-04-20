# -*- coding: utf-8 -*-
"""
Created on 20211018 18:00

@author: mshbing
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime
import collections
import os
import argparse
import json
from json import JSONEncoder
from _ctypes import PyObj_FromPtr
import re
import sys


# def get_contract(date,value,ratio):
#     """
#     from convert_data get new contract of openinsert largest
    
#     Return dic {"IF":"IF2109","IF1":"IF2110"}
#     """
    
#     with open(sym_info_path, "r") as f:
#         symbol_info = json.load(f)

#     dic = collections.OrderedDict()
#     for path in [comdty_path, cffex_path]:
#         file_path = path + 'dominant_contract.csv'
#         df = pd.read_csv(file_path,sep='\t', index_col=0, header=[0,1])
#         for tup in df.columns.tolist():
#             multi = symbol_info[tup[0]]['multiplier']
#             con = df.loc[date, tup]
#             dfc = pd.read_csv(path+con+'.csv', delimiter='\t', index_col=0)
#             close = dfc.loc[date, 'close']
#             max_size = []
#             max_hand = max(1.0, value*10000/(close*multi))
#             max_hand_1 = max(1.0, max_hand/ratio)
#             max_size.append(max_hand)
#             max_size.append(max_hand_1)
#             if tup[1][0] == 'd':
#                 dic[tup[0]] = max_size
#             else:
#                 dic[tup[0]+'1'] = max_size
    
#     return dic

def get_contract(exchange,date,value,ratio):
    """
    from convert_data get new contract of openinsert largest
    
    Return dic {"IF":"IF2109","IF1":"IF2110"}
    """
    file_path = '/shared/shared_data/convert_data.'+exchange+'.day.csv'
    with open(sym_info_path, "r") as f:
        symbol_info = json.load(f)

    df = pd.read_csv(file_path,sep='\t')
    
    if len(df[df['date']==date])>0:
        df_today = df[df['date']==date]
        symbol_list = df_today['symbol'].unique()
        dic = collections.OrderedDict()
        for i in symbol_list:
            symbol = re.findall("[A-Za-z]+", i)[0] 
            multi = symbol_info[symbol]['multiplier']
            # print symbol, multi
            close = df_today[df_today['symbol']==i]['close'].values[-1]
            max_size = []
            max_hand = max(1.0, value*10000/(close*multi))
            max_hand_1 = max(1.0, max_hand/ratio)
            max_size.append(max_hand)
            max_size.append(max_hand_1)
            dic[i] = max_size

    else:
        # print exchange+"today has no data,please check convert data!"
        return None

    return dic


def account_info(account_risk,account_list,extra_account):
    with open(acct_info_path, "r") as g:
        account_conf = json.load(g)

    cmd_brokers = list()
    running_brokers = {}
    product_extra_capital = {}
    for account in account_list:
        # add cmd_brokers
        cmd_brokers.append(account)

        # add running_brokers
        account_dic = {}
        account_dic['product'] = account_conf[account]['product']
        account_dic['capital_past'] = account_conf[account]['capital_past']
        account_dic['multiplier'] = account_conf[account]['multiplier']
        account_dic['stop_loss'] = account_conf[account]['stop_loss']
        account_dic['last_update'] = account_conf[account]['last_update']
        account_dic['warning'] = account_conf[account]['warning']
        account_dic['settlement_past'] = account_conf[account]['settlement_past']
        account_dic['south'] = account_conf[account]['south']
        running_brokers[account] = account_dic

        # add product_extra_capital
        product_extra_capital[account_dic['product']] = account_conf[account]['product_extra_capital']

    for account in extra_account:
        # add running_brokers
        account_dic = {}
        account_dic['product'] = account_conf[account]['product']
        account_dic['capital_past'] = account_conf[account]['capital_past']
        account_dic['multiplier'] = account_conf[account]['multiplier']
        account_dic['stop_loss'] = account_conf[account]['stop_loss']
        account_dic['last_update'] = account_conf[account]['last_update']
        account_dic['warning'] = account_conf[account]['warning']
        account_dic['settlement_past'] = account_conf[account]['settlement_past']
        account_dic['south'] = account_conf[account]['south']
        running_brokers[account] = account_dic

        # add product_extra_capital
        product_extra_capital[account_dic['product']] = account_conf[account]['product_extra_capital']

    # print cmd_brokers
    account_risk['cmd_brokers'] = cmd_brokers
    account_risk['running_brokers'] = running_brokers
    account_risk['product_extra_capital'] = product_extra_capital
    return account_risk


if __name__ == "__main__":
    # PARSER = argparse.ArgumentParser()
    # PARSER.add_argument('-value', help='100',required=False)
    # PARSER.add_argument('-account', help='GXWX1 sym SJTR8',nargs='+', required=False)
    # PARSER.add_argument('-ratio', help='最大持仓与单次最大下单手数的比例', required=False)
    # ARGS = PARSER.parse_args()

    work_dir = '/shared/aqs_trade_conf/'
    acct_info_path = work_dir + "account_info.json"
    sym_info_path = "/shared/database/symbol_info"
    server_info_path = work_dir + 'server_info.json'    

    exchange_list = ['shfe','dce','zce']

    today = datetime.now()
    date = today.year*10000 + today.month*100+today.day


    with open(server_info_path, "r") as f:
        server_info = json.load(f)
    # server_info = {'ZS': {'account': ['SJSX', 'RDQY1', 'ZSWX2', 'SJWX1', 'ZSQY2'], 'value': 100, 'ratio': 3},
    #                 'SJ': {'account': ['GXWX1'], 'value': 25, 'ratio': 1}}

    # if os.path.exists(work_dir+str(date)+'_error.log'):
    #     sys.exit()    

    for server, info in server_info.iteritems():
        conf_path = work_dir+server+'/account_risk_conf.json'
        hist_conf_path = work_dir+server+'/conf_history/account_risk_conf_'+str(date)+'.json'
        account = info['account']
        value = info['value']
        ratio = info['ratio']


         #TUDO cffex tick_data not in 27,need check by human 
        dic_all = {}
        for exchange in exchange_list:
            dic = get_contract(exchange, date, value, ratio)
            dic_all.update(dic)

        # print dic_all

        account_risk = {}
        account_risk['max_size'] = dic_all
        if server == 'ZS':
            extra = ['SJSX']
        else:
            extra = []
        account_risk = account_info(account_risk, account, extra)

        with open(conf_path,'w') as w:
            json.dump(account_risk, w, indent=4)
        with open(hist_conf_path,'w') as w:
            json.dump(account_risk, w, indent=4)
        try:
            os.chmod(hist_conf_path, 0o0777)
        except OSError:
            continue

