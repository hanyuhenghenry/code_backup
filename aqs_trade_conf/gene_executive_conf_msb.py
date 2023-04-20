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


def get_contract(date):
    """
    from convert_data get new contract of openinsert largest
    
    Return dic {"IF":"IF2109","IF1":"IF2110"}
    """
    dic = collections.OrderedDict()
    for exchange in exchange_list:
        file_path = '/shared/shared_data/convert_data.'+exchange+'.day.csv'
        df = pd.read_csv(file_path,sep='\t')
        # print df.tail()
        df_today = df[df['date']==date]
        if not df_today.empty:
            symbol_list = df_today['symbol'].unique()
            for i in symbol_list:
                contract = df_today[df_today['symbol']==i]['contract'].values[0]
                dic[contract] = i 
        else:
            # print exchange+"today has no data,please check convert data!"
            with open(work_dir+str(date)+'_error.log', 'a') as f:
                f.writelines('Convert data is not updated for %s. \n' % exchange)
            continue
    return dic

# def get_contract_wh(exchange,date):
#     """"
#     from tick data get new contract of openinsert largest same to wenhua

#     Return dic {"IF2109":"IF","IC2109":"IC"}
    
#     """
#     tick_path = '/shared/database_comdty/proto_'+exchange+'/'+exchange+'_day/'+exchange+'_'+str(date)+'.h5'
#     tick_data = pd.read_hdf(tick_path)
#     tick_data.rename(columns={'symbol':'contract'},inplace=True)
#     tick_data['symbol'] = tick_data['contract'].str.replace('\d+', '')
#     symbol_list = tick_data['symbol'].unique()
#     # dic = collections.OrderedDict()
#     dic={}
#     for i in symbol_list:
#         df = tick_data[tick_data['symbol']==i].tail(2000)
#         contract_list = df['contract'].unique()
#         opi = 0
#         vol = 0
#         main_contract = None
#         for j in contract_list:
#             df_contract = df[df['contract']==j].tail(1)
#             if (opi == 0) & (vol==0):
#                 opi = df_contract['openinterest'].values[0]
#                 vol = df_contract['volume'].values[0]
#                 main_contract = j
#             else:
#                 if (df_contract['openinterest'].values[0] > opi) & (df_contract['volume'].values[0]>vol): 
#                     opi = df_contract['openinterest'].values[0]
#                     vol = df_contract['volume'].values[0]
#                     main_contract = j
#         dic[main_contract] = i
#     return dic

def update_list_contract(executive_conf,dic_con):
    """
    
    """
    # exchange_list = ['dce']
    list_contract = collections.OrderedDict()
    for key, value in dic_con.items():
        symbol = re.sub(pattern=r"\d", repl=r"", string=key)
        contract_info = []
        contract_info.append(value)
        best_t = best_time[symbol] if symbol in best_time.keys() else  best_time['default']
        limit_t = limit_time[symbol] if symbol in limit_time.keys() else  limit_time['default']
        contract_info.append(best_t)
        contract_info.append(limit_t)
        list_contract[key] = contract_info
        # if exchange != 'cffex':
        #     dic_wh = get_contract_wh(exchange,date)
        #     different_num = 0
        #     for contract, symbol in dic_wh.items():
        #         if contract not in dic_con.keys():
        #             print contract + ":  be different from wenhua,pleace check it"
        #         elif dic_con[contract] != symbol:
        #             different_num += 1
        #             print contract + ":  be different from wenhua,pleace check it"
        #     if different_num==0:
        #         print exchange+":main contract are all same to wenhua"
    # print list_contract
    executive_conf['list_contract'] = list_contract
    return executive_conf


def update_address(executive_conf,executive_info):
    executive_conf['frontend_address'] = executive_info['frontend_address']
    executive_conf['backend_address'] = executive_info['backend_address']
    return executive_conf


def update_close_method(executive_conf,executive_info):
    close_method = {}
    close_method['today_prior'] = executive_info['close_method']['today_prior']
    close_method['open_opposite'] = executive_info['close_method']['open_opposite']
    close_method['yesterday_prior'] = executive_info['close_method']['yesterday_prior']
    executive_conf['close_method'] = close_method
    return executive_conf 


def update_order_manager(executive_conf,executive_info):
    order_manager = {}
    order_manager['tick_multiplier'] = executive_info['order_manager']['tick_multiplier']
    order_manager['tick_size'] = executive_info['order_manager']['tick_size']
    order_manager['market_num'] = executive_info['order_manager']['market_num']
    order_manager['market_time'] = executive_info['order_manager']['market_time']
    order_manager['tick_num'] = executive_info['order_manager']['tick_num']
    executive_conf['order_manager'] = order_manager
    return executive_conf


def update_flatten(executive_conf,executive_info,server):
    # if server=='SJ':
    #     executive_conf['flatten'] = executive_info['flatten']['sj']
    # elif server == 'ZS':
    #     executive_conf['flatten'] = executive_info['flatten']['zs']
    # return executive_conf
    executive_conf['flatten'] = executive_info['flatten'][server.lower()]
    return executive_conf


def update_parameter(executive_conf,executive_info):
    executive_conf['parameter'] = executive_info['parameter']
    return executive_conf

def update_product_margin(executive_conf,executive_info):
    executive_conf['product_margin'] = executive_info['product_margin']
    return executive_conf

def contract_update(dic_con):
    try:
        with open(conf_path, "r") as f:
            prev_conf = json.load(f)
    except IOError:
        return []
    prev_dic_con = prev_conf['list_contract']
    dct = {v: k for k, v in dic_con.items()}
    prev_dct = {v[0]: k for k, v in prev_dic_con.iteritems()}
    change = []
    for k, v in prev_dct.iteritems(): 
        if k not in dct.keys():
            change.append([k, v, ''])
        elif v != dct[k]:
            change.append([k, v, dct[k]])
    return change


if __name__ == "__main__":
    # PARSER = argparse.ArgumentParser()
    # PARSER.add_argument('-server', help='sj',required=False)
    # ARGS = PARSER.parse_args()
    work_dir = '/shared/aqs_trade_conf/'
    exec_info_path = work_dir + 'executive_info.json'
    server_info_path = work_dir + 'server_info.json'

    exchange_list = ['shfe','dce','zce','cffex'] #TUDO cffex tick_data not in 27,need check by human 

    with open(server_info_path, "r") as f:
        server_list = json.load(f).keys()
    today = datetime.now()
    date = today.year*10000+today.month*100+today.day

    # best_time={"default": 0.5,"CF": 60,"PF": 600,"SR": 900,"TA": 600,"UR": 900,"ZC": 900,"sn": 60,"ni": 120,
    #     "zn": 300,"ag": 600,"bu": 900,"sp": 900,"eb": 120,"fu": 900,"hc": 600,"rb": 900,"i": 900,"j": 900,"jm": 600,"sc": 300,}
    best_time={"default": 0.5,"i": 900,"rb": 900,"MA":30, "p":30,"y":30, "OI":30, "c":30, "m":30, "TA":60, "RM":60, "CF":60, "ru":60, "SR":900}

    inactive_symbol = ['']
    limit_time = {"default":0}

    with open(exec_info_path, "r") as f:
        executive_info = json.load(f)

    dic_con = get_contract(date)
    if not dic_con:
        sys.exit()

    change = []

    for server in server_list:
        conf_path = work_dir+server+'/executive_conf.json'
        hist_conf_path = work_dir+server+'/conf_history/executive_conf_'+str(date)+'.json'

        if server == server_list[0]:
            change = contract_update(dic_con)

        executive_conf = {}

        execuive_conf = update_list_contract(executive_conf,dic_con)
        execuive_conf = update_address(executive_conf,executive_info)
        execuive_conf = update_close_method(executive_conf,executive_info)
        execuive_conf = update_order_manager(executive_conf,executive_info)
        execuive_conf = update_flatten(executive_conf,executive_info,server)
        execuive_conf = update_parameter(executive_conf,executive_info)
        execuive_conf = update_product_margin(executive_conf,executive_info)

        with open(conf_path,'w') as f:
            json.dump(executive_conf, f, indent=4)
        with open(hist_conf_path,'w') as f:
            json.dump(executive_conf, f, indent=4)
        try:
            os.chmod(hist_conf_path, 0o0777)
        except OSError:
            continue
    
    if change:
        with open(work_dir+'contract_change.log', 'a') as f:
            for c in change:
                if c[2] == '':
                    f.writelines('%s: %s not in convert data today. \n' % (str(date), c[0]))
                else:
                    f.writelines('%s: %s changes from %s to %s. \n' % tuple([str(date)]+c))
