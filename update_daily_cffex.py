import glob
import os
import pandas as pd
import numpy as np
import json
import re
import datetime
import sys
import logging
from aqs_flow.futures_data import FuturesData
futures = FuturesData(host='192.168.1.33')
# from aqs_data import futures
from collections import defaultdict

def get_contract(x, date):
    a = dfcon[(dfcon['date']==date)&(dfcon['symbol']==x)]['contract']
    if not a.empty:
        return a.iloc[0]
    if x[-1] != '1':
        return dfmain.loc[date, x]['dominant']
    else:
        return dfmain.loc[date, x[:-1]]['sub_domi'] 

def to_minute():
    ends = defaultdict(list)
    for s in syms:
        e = sym_info[s]['trade_times'][-1]['end_time']
        ends[e].append(s)
    ret = []
    for e, sym in ends.items():
        end = datetime.time(hour=e//100, minute=e%100).strftime('%H:%M')
        sym1 = list(map(lambda x: x+'1', sym))
        con = list(map(lambda x: get_contract(x, date), sym))
        con1 = list(map(lambda x: get_contract(x, date), sym1))
        data = read_data(con, e)
        dfm = tick_to_min(data, end, 0)
        ret.append(dfm)
        data = read_data(con1, e)
        dfm = tick_to_min(data, end, 1)
        ret.append(dfm)
    df = pd.concat(ret)
    df = df.sort_values(['symbol', 'timestamp_exchange'])
    return df
        
def tick_to_min(df, end, status):
    dfm = df.groupby(['contract', 'time'], sort=False)['lastprice'].ohlc()
    dfm['volume'] = df.groupby(['contract', 'time'], sort=False)['tick_volume'].sum()
    dfm['turnover'] = df.groupby(['contract', 'time'], sort=False)['tick_turnover'].sum()
    dfm['total_avg'] = df.groupby(['contract', 'time'], sort=False)['total_average'].last()
    dfm = dfm.reset_index(level=0)
    skeleton = pd.DataFrame(index=pd.date_range(str(date)+' 09:30', str(date)+' '+end, freq='1min', closed='left'))
    skeleton = skeleton.between_time('13:00', '11:29')
    skeleton['time'] = skeleton.index.strftime('%H:%M')
    skeleton = skeleton.reset_index()
    res = []
    for contract in dfm['contract'].unique():
        dfs = dfm[dfm['contract'] == contract]
        dfs = skeleton.merge(dfs, on='time', how='left')
        dfs['contract'] = dfs['contract'].fillna(contract)
        dfs[['close', 'total_avg']] = dfs[['close', 'total_avg']].fillna(method='ffill')
        dfs['volume'] = dfs['volume'].fillna(0)
        dfs['turnover'] = dfs['turnover'].fillna(0)
        for col in ['open', 'high', 'low']:
            dfs[col] = dfs[col].fillna(dfs['close'])
        if not status:
            dfs['symbol'] = re.sub('[^a-zA-Z]+', '', contract)
        else:
            dfs['symbol'] = re.sub('[^a-zA-Z]+', '', contract) + '1'
        dfs['date'] = date
        dfs = dfs.rename(columns={'index': 'timestamp_exchange'})
        dfs = dfs[eff_cols]
        res.append(dfs)
    dfm = pd.concat(res)
    return dfm

def read_data(contract_list, end):
    # df = futures.get_contract_today_tick(contract_list, fields=useful_fields)
    df = futures.get_contract_his_tick(date, date, contract_list, useful_fields)
    # df = futures.get_contract_his_tick(contract_list, date, date, useful_fields)
    df.loc[df.time == 929, 'time'] = 930
    df.loc[df.time == 1130, 'time'] = 1129
    df = df[df['tick_volume'] != 0]
    df = df.rename(columns={'instr_id': 'contract'})
    df['time'] = df['time'].apply(lambda x: datetime.time(hour=x//100, minute=x%100)).astype(str).str.slice(0, -3)
    return df


def to_sym(df):
    for sym in df['symbol'].unique():
        dfc = df[df['symbol'] == sym]
        dfc.to_csv(write_dir_sym+sym, sep='\t', mode='a', index=False, header=False)


def to_30min(df):
    df = df.set_index('timestamp_exchange')
    df.index = pd.to_datetime(df.index)
    df['ret'] = np.log(df['close'] / df['open'])
    df = df.merge(df.groupby('symbol')['ret'].expanding().mean().reset_index().rename(columns={'ret': 'mean'}), on=['timestamp_exchange', 'symbol'], how='left').set_index('timestamp_exchange')
    df = df.merge(df.groupby('symbol')['ret'].expanding().std().reset_index().rename(columns={'ret': 'std'}), on=['timestamp_exchange', 'symbol'], how='left').set_index('timestamp_exchange')
    df['above'] = df['close'] >= df['total_avg']
    df['prev_above'] = df.groupby('symbol')['above'].shift(1)
    df['prev_above'] = df['prev_above'].fillna(df['above'])
    df['cross'] = (df['above'] != df['prev_above']).astype(int)
    df.index = pd.to_datetime(df.index)
    dflst = []
    for t in ['I', 'T']:
        dfd = df[df['symbol'].str.slice(0,1) == t]
        if t == 'I':
            checkpoints = checkpointsI
        elif date < 20200720:
            checkpoints = ['09:15:00'] + checkpointsT
        else:
            checkpoints = ['09:30:00'] + checkpointsT          
        ckpts = pd.to_datetime(list(map(lambda x: str(date)+' '+x, checkpoints)))
        # bins = pd.IntervalIndex.from_arrays(ckpts[:-1], ckpts[1:], 'left')
        # group = dfd.groupby(['symbol', pd.cut(dfd.index, bins)])
        group = dfd.groupby(['symbol', pd.cut(dfd.index, ckpts, right=False)])
        dfd30 = group['open'].first().reset_index()
        if dfd30.dropna().empty:
            raise Exception
        dfd30['high'] = group['high'].max().reset_index()['high']
        dfd30['low'] = group['low'].min().reset_index()['low']
        dfd30['close'] = group['close'].last().reset_index()['close']
        dfd30['volume'] = group['volume'].sum().reset_index()['volume']
        dfd30['turnover'] = group['turnover'].sum().reset_index()['turnover']
        dfd30['contract'] = group['contract'].first().reset_index()['contract']
        dfd30['cmean'] = group['mean'].last().reset_index()['mean']
        dfd30['cstd'] = group['std'].last().reset_index()['std']
        dfd30['dmean'] = group['ret'].mean().reset_index()['ret']
        dfd30['dstd'] = group['ret'].std().reset_index()['ret']
        dfd30['cross'] = group['cross'].sum().reset_index()['cross']
        dfd30['date'] = date
        dfd30['timestamp_exchange'] = pd.to_datetime(dfd30['level_1'].apply(lambda x: x.right))
        dfd30['time'] = dfd30['timestamp_exchange'].dt.strftime('%H:%M')
        dfd30 = dfd30.drop(columns=['level_1'])
        dflst.append(dfd30)
    df30 = pd.concat(dflst)
    df30 = df30.sort_values(['symbol', 'timestamp_exchange']).reset_index(drop=True)
    df30['cum_cross'] = df30.groupby('symbol')['cross'].cumsum().reset_index()['cross']
    df30['csharpe'] = df30['cmean'] / df30['cstd']
    df30['dsharpe'] = df30['dmean'] / df30['dstd']
    df30['adj_csharpe'] = df30['csharpe'] / (df30['cum_cross'] + 1)
    df30['adj_dsharpe'] = df30['dsharpe'] / (df30['cross'] + 1)
    df30 = df30.set_index('timestamp_exchange')
    df30 = df30.fillna(0.0)
    return df30

def add_rank():
    ex = 'cffex'
    df = pd.read_csv(write_dir_30min+ex+'.csv', delimiter='\t')
    df = df.groupby('symbol').tail(250).reset_index(drop=True)
    df['abs_csharpe'] = np.abs(df['csharpe'])
    df['abs_dsharpe'] = np.abs(df['dsharpe'])
    df['abs_adj_csharpe'] = np.abs(df['adj_csharpe'])
    df['abs_adj_dsharpe'] = np.abs(df['adj_dsharpe'])
    group = df.groupby('symbol')
    df['crank'] = group['abs_csharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_csharpe']]
    df['drank'] = group['abs_dsharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_dsharpe']]
    df['adj_crank'] = group['abs_adj_csharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_adj_csharpe']]
    df['adj_drank'] = group['abs_adj_dsharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_adj_dsharpe']]
    df = df.drop(columns=['cmean', 'dmean', 'cstd', 'dstd', 'abs_csharpe', 'abs_dsharpe'])
    df = df.drop(columns=['abs_adj_csharpe', 'abs_adj_dsharpe'])
    df = df[df['date'] == date]
    return df

def calculate_sig(df): #cffex
    day1 = -1
    day2 = -1
    dfd = df[(df['time']>='09:31') & (df['time']<='09:40')]
    bm = dfd[dfd['time']=='09:31']['volume'].iloc[0]
    dfd['vol_flag'] = (dfd['volume']>bm).astype(int)
    day2 = dfd['vol_flag'].sum()
    day1 = dfd[dfd['time']<='09:35']['vol_flag'].sum()
    idx = ['early_simple1', 'early_simple2']
    return pd.Series((day1, day2), index=idx)

def early_simple(df):
    dff = df[df['symbol'].isin(['IC', 'IF', 'IH', 'IM'])]
    group = dff.groupby(['date', 'symbol'])
    dfg = group.apply(calculate_sig).reset_index()
    dfg = dfg.sort_values(['date', 'symbol'])
    return dfg

if __name__ == '__main__':
    database_dir = '/shared/database_cffex/'
    write_dir_date = '/shared/shared_data/cffex_data/1min_by_date/'
    write_dir_sym = '/shared/shared_data/cffex_data/1min_by_sym/'
    write_dir_30min = '/shared/shared_data/cffex_data/'
    write_dir_sha = '/shared/strategy_stats/'
    write_dir_es = '/shared/strategy_stats/early_simple_cffex.csv'
    work_dir = '/home/hyh/resample_data/'
    sym_path = '/shared/database/Product_info.json'
    with open(sym_path,'r') as f:
        sym_info = json.loads(f.read())
    convert_dir = '/shared/shared_data/'
    dfmain = pd.read_csv('/shared/shared_data/cffex_data/daily/dominant_contract.csv', delimiter='\t', index_col=0, header=[0,1])
    dfcon = pd.read_csv(convert_dir+'convert_data.cffex.day.csv', delimiter='\t')
    calendar = pd.read_csv('/shared/database/shfe_trading.calendar', delimiter='\t', index_col='date')
    date = int(datetime.datetime.today().date().strftime('%Y%m%d'))
    syms = ['IF', 'IC', 'IH', 'T', 'TS', 'TF', 'IM']
    useful_fields = ['instr_id', 'trading_day', 'action_day', 'time', 'second', 'milli_sec', 'lastprice', 'tick_volume', 'tick_turnover', 'total_average']
    eff_cols = ['timestamp_exchange', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'symbol', 'contract', 'date', 'time', 'total_avg']

    mlist = []
    for s in syms:
        if s[-1] == '1':
            continue
        try:
            m = sym_info[s]['multiple']
        except KeyError:
            continue
        mlist.append([s, m])
        mlist.append([s+'1', m])
    mult = pd.DataFrame(mlist, columns=['symbol', 'mult'])

    checkpointsI = ['09:30:00', '10:00:00', '10:30:00', '11:00:00', '11:30:00','13:30:00','14:00:00','14:30:00','15:00:00']
    checkpointsT = ['09:45:00', '10:15:00', '10:45:00', '11:15:00','13:15:00','13:45:00','14:15:00','14:45:00','15:15:00']

    log_file = work_dir+str(date)+"cffex_error.log"
    # logging.basicConfig(filename=log_file, filemode='a', format='%(levelname)s %(asctime)s - %(message)s', level=logging.ERROR)
    logger = logging.getLogger('cffex')
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(log_file, 'a')
    handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s - %(message)s'))
    logger.addHandler(handler)

    output_path = write_dir_date + str(date)
    try:
        df = to_minute()
    except:
        logger.error('Something wrong when converting '+str(date)+' to minute data')
        sys.exit()
    df.to_csv(output_path, sep='\t', mode='a', index=False, header=not os.path.exists(output_path))
    to_sym(df)
    try:
        df30 = to_30min(df)
    except:
        logger.error('Something wrong when converting '+str(date)+' to 30 minute data')
        sys.exit()
    df30.to_csv(write_dir_30min+'cffex.csv', sep='\t', mode='a', header=False)
    try:
        dfes = early_simple(df)
    except:
        logger.error('Something wrong when converting '+str(date)+' to early simple data')
        sys.exit()
    dfes.to_csv(write_dir_es, sep='\t', mode='a', header=False, index=False)
    try:
        dfsha = add_rank()
    except:
        logger.error('Something wrong when adding sharpe rank to '+str(date)+' data')
        sys.exit()
    dfsha.to_csv(write_dir_sha+'cffex_sha.csv', sep='\t', mode='a', header=False, index=False)

    if os.path.exists(log_file) and os.stat(log_file).st_size == 0:
        os.remove(log_file)
