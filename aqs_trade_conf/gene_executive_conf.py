# -*- coding: utf-8 -*-
"""
Created on 20211018 18:00

@author: mshbing
"""
import pandas as pd
import numpy as np
import time
import datetime
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


def get_contract_jq(date):
    """
    from convert_data get new contract of openinsert largest
    
    Return dic {"IF":"IF2109","IF1":"IF2110"}
    """
    dic = collections.OrderedDict()
    for exchange in exchange_list:
        if exchange == 'cffex':
            continue
        file_path = '/shared/shared_data/convert_jq.'+exchange+'.day.csv'
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
                f.writelines('Convert jq is not updated for %s. \n' % exchange)
            continue
    return dic

# def get_contract(date):
#     """
#     from convert_data get new contract of openinsert largest
    
#     Return dic {"IF":"IF2109","IF1":"IF2110"}
#     """
#     dic = collections.OrderedDict()
#     for path in [comdty_path, cffex_path]:
#         file_path = path + 'dominant_contract.csv'
#         df = pd.read_csv(file_path,sep='\t', index_col=0, header=[0,1])
#         for tup in df.columns.tolist():
#             try:
#                 con = df.loc[date, tup]
#             except KeyError:
#                 with open(work_dir+str(date)+'_error.log', 'a') as f:
#                     f.writelines('Main contract info is not updated. \n')
#                 sys.exit()
#             if tup[1][0] == 'd':
#                 dic[con] = tup[0]
#             else:
#                 dic[con] = tup[0] + '1'
#     return dic


# def update_list_contract(server,excutive_conf,dic_con):
#     if server == 'SJ':
#         return update_list_contract_old(excutive_conf, dic_con)
#     elif server == 'ZS':
#         return update_list_contract_new(executive_conf, dic_con)

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
        limit_t = defaults['limit_time']
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

# def update_list_contract_new(executive_conf, dic_con):
#     list_contract = collections.OrderedDict()
#     for key, value in dic_con.items():
#         symbol = re.sub(pattern=r"\d", repl=r"", string=key)
#         if symbol not in active_symbols:
#             continue
#         contract_info = {}
#         contract_info['alias'] = value
#         time = [0, 0, 0]
#         time[0] = best_time[symbol] if symbol in best_time.keys() else best_time['default']
#         time[1] = time_param[symbol]['time_to_super_1']
#         time[2] = limit_time[symbol] if symbol in limit_time.keys() else limit_time['default']
#         contract_info['time'] = time
#         super_dct = {}
#         super_dct['ratio'] = time_param[symbol]['ratio']
#         super_dct['up_mean_mean_2'] = tick_needs.loc[symbol, 'up_mean_mean']
#         if tick_needs.loc[symbol, 'active']:
#             super_dct['up_mean_mean_1'] = super_dct['up_mean_mean_2'] / 2.0 + 1.0
#         else:
#             super_dct['up_mean_mean_1'] = super_dct['up_mean_mean_2'] / 2.0 + 2.0
#         super_dct['vol_cut'] = threshold[key][0]
#         super_dct['oi_cut'] = threshold[key][1]
#         super_dct['tick_num'] = tick_num[symbol] if symbol in tick_num.keys() else tick_num['default']
#         super_dct['delay'] = delay[symbol] if symbol in delay.keys() else delay['default']
#         super_dct['update_interval'] = time_param[symbol]['time_to_super_2']
#         contract_info['super'] = super_dct
#         list_contract[key] = contract_info
#         # if exchange != 'cffex':
#         #     dic_wh = get_contract_wh(exchange,date)
#         #     different_num = 0
#         #     for contract, symbol in dic_wh.items():
#         #         if contract not in dic_con.keys():
#         #             print contract + ":  be different from wenhua,pleace check it"
#         #         elif dic_con[contract] != symbol:
#         #             different_num += 1
#         #             print contract + ":  be different from wenhua,pleace check it"
#         #     if different_num==0:
#         #         print exchange+":main contract are all same to wenhua"
#     # print list_contract
#     executive_conf['list_contract'] = list_contract
#     return executive_conf

def update_twap(executive_conf):
    list_contract = executive_conf['list_contract']
    dct = {}
    for k in list_contract.keys():
        dct[k] = []
        sym = list_contract[k][0]
        dct[k].append(sym)
        dct[k].append(twap_time[sym] if sym in twap_time.keys() else twap_time['default'])
        dct[k].append(twap_interval[sym] if sym in twap_interval.keys() else twap_interval['default'])
        dct[k].append(vt[sym] if sym in vt.keys() else vt['default'])
        dct[k].append(vm[sym] if sym in vm.keys() else vm['default'])
    executive_conf['twap_params'] = dct
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


def update_order_manager(executive_conf,executive_info, server):
    order_manager = {}
    order_manager['tick_multiplier'] = executive_info['order_manager']['tick_multiplier'][server.lower()]
    order_manager['tick_size'] = executive_info['order_manager']['tick_size']
    order_manager['market_num'] = executive_info['order_manager']['market_num']
    order_manager['market_time'] = executive_info['order_manager']['market_time']
    order_manager['tick_num'] = executive_info['order_manager']['tick_num']
    order_manager['least_open'] = update_least_open()
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

def update_least_open():
    if not least_open:
        return {}
    else:
        least_open_con = {}
        for k, v in least_open.iteritems():
            if k in dic_con_re:
                least_open_con[dic_con_re[k]] = v
        return least_open_con

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

def update_open_params(executive_conf):
    lst = open_multi.index.tolist()
    omcon = {}
    for k, con in dic_con_re.iteritems():
        sym = filter(str.isalpha, con)
        if sym in open_multi.index.tolist():
            multi = open_multi.loc[sym, 'multi']
        else:
            multi  = 1
        if sym in open_regular_time.keys():
            wait_time = open_regular_time[sym][0]
            twap_time = open_regular_time[sym][1]
        else:
            wait_time = 60
            twap_time = 30
        omcon[con] = [multi, wait_time, twap_time]
    executive_conf['open_params'] = omcon
    return executive_conf

def update_end_time(executive_conf):
    dct = executive_conf['twap_params'].copy()
    new_dct = {}
    for k, v in dct.iteritems():
        if v[0] in am1_end or v[0][:-1] in am1_end:
            end_time = '00:59:50'
        elif v[0] in am230_end or v[0][:-1] in am230_end:
            end_time = '02:29:50'
        else:
            end_time = '22:59:50'
        e1 = datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.timedelta(seconds=v[1])
        e2 = datetime.datetime.strptime('14:59:50', '%H:%M:%S') - datetime.timedelta(seconds=v[1])
        e1 = e1.strftime('%H:%M:%S')
        e2 = e2.strftime('%H:%M:%S')
        new_dct[k] = [e1, e2]
    executive_conf['end_time'] = new_dct
    return executive_conf

def update_close_params(executive_conf):
    dct = {}
    for k, con in dic_con_re.iteritems():
        sym = filter(str.isalpha, con)
        if sym in close_twap_time.keys():
            ttime = close_twap_time[sym]
        else:
            ttime = 30
        e1 = datetime.datetime.strptime('22:59:50', '%H:%M:%S') - datetime.timedelta(seconds=ttime)
        e2 = datetime.datetime.strptime('14:59:50', '%H:%M:%S') - datetime.timedelta(seconds=ttime)
        e1 = e1.strftime('%H:%M:%S')
        e2 = e2.strftime('%H:%M:%S')
        dct[con] = [e1, e2, ttime]
    executive_conf['close_params'] = dct
    return executive_conf
    

if __name__ == "__main__":
    # PARSER = argparse.ArgumentParser()
    # PARSER.add_argument('-server', help='sj',required=False)
    # ARGS = PARSER.parse_args()
    work_dir = '/shared/aqs_trade_conf/'
    exec_info_path = work_dir + 'executive_info.json'
    server_info_path = work_dir + 'server_info.json'
    tick_path = work_dir + 'tick_needs/tick_needs.csv'
    threshold_path = work_dir + 'threshold.json'
    time_param_path = work_dir + 'super_param/super_param.json'
    limit_time_path = work_dir + 'limit_time.json'
    twap_time_path = work_dir + 'twap_time.json'
    least_open_path = work_dir + 'least_open.json'
    open_multi_path = work_dir + 'open_multi.csv'
    open_regular_time_path = work_dir + 'open_regular_time.json'
    vthresh_path = work_dir + 'accelerate_vthresh.csv'
    close_twap_time_path = work_dir + 'close_twap_time.json'
    comdty_path = '/shared/shared_data/comdty_data/daily/'
    cffex_path = '/shared/shared_data/cffex_data/daily/'

    defaults =  {'limit_time': 0, 'tick_num': 5, 'delay': 1}
    # DEFAULTS; CHANGE
    tick_num = {'default': 5}
    delay = {'default': 1}
    # garbage = ['rr', 'fb', 'wr', 'WH', 'RS', 'LR', 'RI', 'bb', 'JR']
    exchange_list = ['shfe','dce','zce','cffex'] #TUDO cffex tick_data not in 27,need check by human 

    am1_end = ['al', 'cu', 'ni', 'zn', 'pb', 'sn', 'ss']
    am230_end = ['ag', 'au', 'sc', 'bc']

    # best_time={"default": 0.5,"CF": 60,"PF": 600,"SR": 900,"TA": 600,"UR": 900,"ZC": 900,"sn": 60,"ni": 120,
    #     "zn": 300,"ag": 600,"bu": 900,"sp": 900,"eb": 120,"fu": 900,"hc": 600,"rb": 900,"i": 900,"j": 900,"jm": 600,"sc": 300,}
    # best_time={"default": 0.5,"i": 900,"rb": 900,"MA":30, "p":30,"y":30, "OI":30, "c":30, "m":30, "TA":60, "RM":60, "CF":60, "ru":60, "SR":900}

    # inactive_symbol = ['']
    # limit_time = {"default":0}

    with open(server_info_path, "r") as f:
        server_list = json.load(f).keys()
    today = datetime.datetime.now()
    date = today.year*10000+today.month*100+today.day

    # if time.time()-os.path.getmtime(tick_path) > 24*60*60:
    #     with open(work_dir+str(date)+'_error.log', 'a') as f:
    #         f.writelines('Warning: tick_needs.csv has not been updated today! \n')
    # tick_needs = pd.read_csv(tick_path, delimiter='\t', index_col=0)

    # if time.time()-os.path.getmtime(threshold_path) > 24*60*60:
    #     with open(work_dir+str(date)+'_error.log', 'a') as f:
    #         f.writelines('Warning: threshold.json has not been updated today! \n')
    # with open(threshold_path, "r") as f:
    #     threshold = json.load(f)

    # if time.time()-os.path.getmtime(time_param_path) > 24*60*60:
    #     with open(work_dir+str(date)+'_error.log', 'a') as f:
    #         f.writelines('Warning: super_param.json has not been updated today! \n')
    with open(time_param_path, "r") as f:
        time_param = json.load(f)

    with open(limit_time_path, 'r') as f:
        limit_time = json.load(f)['limit_time']
    limit_time['default'] = 60

    with open(twap_time_path, 'r') as f:
        twap_time = json.load(f)
    twap_time['default'] = 30

    with open(least_open_path, 'r') as f:
        least_open = json.load(f)

    with open(open_regular_time_path, 'r') as f:
        open_regular_time = json.load(f)

    with open(close_twap_time_path, 'r') as f:
        close_twap_time = json.load(f)
    
    open_multi = pd.read_csv(open_multi_path, delimiter='\t', index_col=0)
    open_multi = open_multi.groupby(open_multi.index).first()

    twap_interval = {}
    twap_interval['default'] = 0.2

    vt = pd.read_csv(vthresh_path, delimiter='\t')
    if vt['date'].iloc[0] != date:
        with open(work_dir+str(date)+'_error.log', 'a') as f:
            f.writelines('Vthresh is not updated today.')
    vt = vt[['symbol', 'vthresh']].set_index('symbol').to_dict()['vthresh']
    vt['default'] = 1000000

    vm = {}
    vm['default'] = 4

    active_symbols = time_param.keys()
    best_time = {k: v['best_to_market_time'] for k,v in time_param.iteritems()}
    best_time['default'] = 0.5


    change = []

    for server in ['SJ', 'msb']:
        conf_path = work_dir+server+'/executive_conf.json'
        hist_conf_path = work_dir+server+'/conf_history/executive_conf_'+str(date)+'.json'
        if server == 'SJ':
            dic_con = get_contract(date)
            if not dic_con:
                sys.exit()
            dic_con_re = {v: k for k, v in dic_con.iteritems()}
        else:
            dic_con = get_contract_jq(date)
            if not dic_con:
                sys.exit()
            dic_con_re = {v: k for k, v in dic_con.iteritems()}

        if server == 'SJ':
            change = contract_update(dic_con)

        executive_conf = {}

        with open(exec_info_path, "r") as f:
            executive_info = json.load(f)

        executive_conf = update_list_contract(executive_conf,dic_con)
        executive_conf = update_twap(executive_conf)
        executive_conf = update_open_params(executive_conf)
        executive_conf = update_end_time(executive_conf)
        executive_conf = update_close_params(executive_conf)
        executive_conf = update_address(executive_conf,executive_info)
        executive_conf = update_close_method(executive_conf,executive_info)
        executive_conf = update_order_manager(executive_conf,executive_info,server)
        executive_conf = update_flatten(executive_conf,executive_info,server)
        executive_conf = update_parameter(executive_conf,executive_info)
        executive_conf = update_product_margin(executive_conf,executive_info)


        with open(conf_path,'w') as f:
            json.dump(executive_conf, f, indent=4)
        with open(hist_conf_path,'w') as f:
            json.dump(executive_conf, f, indent=4)
        try:
            os.chmod(hist_conf_path, 0o0777)
        except OSError:
            continue
         
        # with open('test.json', 'w') as f:
        #    json.dump(executive_conf, f, indent=4)
    
    if change:
        with open(work_dir+'contract_change.log', 'a') as f:
            for c in change:
                if c[2] == '':
                    f.writelines('%s: %s not updated today. \n' % (str(date), c[0]))
                else:
                    f.writelines('%s: %s changes from %s to %s. \n' % tuple([str(date)]+c))

