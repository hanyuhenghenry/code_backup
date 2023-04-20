import glob
import os
import pandas as pd
import numpy as np
import json
import re
import datetime
import sys
import logging
# from aqs_data import futures
from aqs_flow.futures_data import FuturesData
futures = FuturesData()
from collections import defaultdict
# import jqdatasdk as jq
# USE DATA-POOL-ENV 

def get_contract(x, date, dfc, dfcon):
    a = dfcon[(dfcon['date']==date)&(dfcon['symbol']==x)]['contract']
    if not a.empty:
        return a.iloc[0]
    b = dfc[(dfc['date']==date)&(dfc['symbol']==x)]['contract']
    if not b.empty:
        return b.iloc[0]
    if x[-1] != '1':
        return dfmain.loc[date, x]['dominant']
    else:
        return dfmain.loc[date, x[:-1]]['sub_domi'] 

def to_minute(ex):
    syms = symbol_lists[ex]
    ends = defaultdict(list)
    for s in syms:
        e = sym_info[s]['trade_times'][-1]['end_time']
        ends[e].append(s)
    ret = []
    for e, sym in ends.items():
        end = datetime.time(hour=e//100, minute=e%100).strftime('%H:%M')
        con = []
        con1 = []
        for s in sym:
            con.append(get_contract(s, date, dfc, dfcon))
            con1.append(get_contract(s+'1', date, dfc, dfcon))
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
    dfm['opi'] = df.groupby(['contract', 'time'], sort=False)['tick_opi'].sum()
    dfm['total_avg'] = df.groupby(['contract', 'time'], sort=False)['total_average'].last()
    dfm = dfm.reset_index(level=0)
    if df[df['time']>'16:00'].shape[0] <= 10:
        end = '15:00'
    skeleton = pd.DataFrame(index=pd.date_range(str(date)+' 09:00', str(date)+' 15:00', freq='1min', closed='left'))
    skeleton = skeleton.between_time('10:30', '10:14').between_time('13:30', '11:29')
    if end == '23:00':
        yesterday = df['action_day'].min()
        skeleton = pd.DataFrame(index=pd.date_range(str(yesterday)+' 21:00', str(yesterday)+' '+end, freq='1min', closed='left')).append(skeleton)
    elif end != '15:00':
        lst = sorted(df['action_day'].unique())
        skeleton = pd.DataFrame(index=pd.date_range(str(lst[0])+' 21:00', str(lst[1])+' '+end, freq='1min', closed='left')).append(skeleton)
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
        dfs['opi'] = dfs['opi'].fillna(0)
        for col in ['open', 'high', 'low']:
            dfs[col] = dfs[col].fillna(dfs['close'])
        dfs = dfs.dropna()
        if not status:
            dfs['symbol'] = re.sub('[^a-zA-Z]+', '', contract)
        else:
            dfs['symbol'] = re.sub('[^a-zA-Z]+', '', contract) + '1'
        dfs['date'] = date
        dfs = dfs.rename(columns={'index': 'timestamp_exchange'})
        dfs = dfs[eff_cols]
        dfs['pre_close'] = dfs['close'].shift(1)
        dfs['pre_close'].iloc[0] = df['pre_close'].iloc[0]
        res.append(dfs)
    dfm = pd.concat(res)
    return dfm

def read_data(contract_list, end):
    # df = futures.get_contract_today_tick(contract_list, fields=useful_fields)
    df = futures.get_contract_his_tick(date, date, contract_list, fields=useful_fields)
    # df = futures.get_contract_his_tick(contract_list, 20220426, 20220426, useful_fields)
    dfy = futures.get_contract_his_tick(yesterday, yesterday, contract_list, fields=useful_fields)
    df = df.sort_values(['instr_id', 'trading_day', 'action_day', 'time', 'second', 'milli_sec', 'local_ts'])
    dfy = dfy.sort_values(['instr_id', 'trading_day', 'action_day', 'time', 'second', 'milli_sec', 'local_ts'])
    dfy1 = dfy[dfy['time'] == 1500]
    dfy1 = dfy1.groupby('instr_id').first().reset_index()
    dfy = dfy[dfy['time'] < 1500]
    dfy = pd.concat([dfy, dfy1])
    dfy = dfy.groupby('instr_id').last().reset_index()
    df = df.merge(dfy[['instr_id', 'total_opi']], on='instr_id', how='left', suffixes=['', '_pre'])
    df['total_opi_pre'] = df['total_opi_pre'].fillna(df['pre_opi'])
    df['tick_opi'] = df.groupby('instr_id')['total_opi'].diff(1)
    df['tick_opi'] = df['tick_opi'].fillna(df['total_opi']-df['total_opi_pre'])
    df['tick_volume'] = df.groupby('instr_id')['total_volume'].diff(1)
    df['tick_volume'] = df['tick_volume'].fillna(df['total_volume'])
    df['tick_turnover'] = df.groupby('instr_id')['total_turnover'].diff(1)
    df['tick_turnover'] = df['tick_turnover'].fillna(df['total_turnover'])
    df.loc[df.time == 859, 'time'] = 900
    df.loc[df.time == 1015, 'time'] = 1014
    df.loc[df.time == 1130, 'time'] = 1129
    df.loc[df.time == 1500, 'time'] = 1459
    df.loc[df.time == 2059, 'time'] = 2100
    if end == 2300 or end == 100:
        df.loc[df.time == end, 'time'] = end-41
    elif end == 230:
        df.loc[df.time == end, 'time'] = end-1
    df = df[df['tick_volume'] != 0]
    df = df.rename(columns={'instr_id': 'contract'})
    df['time'] = df['time'].apply(lambda x: datetime.time(hour=x//100, minute=x%100)).astype(str).str.slice(0, -3)
    return df

def to_sym(df):
    for sym in df['symbol'].unique():
        dfc = df[df['symbol'] == sym]
        dfc.to_csv(write_dir_sym+sym, sep='\t', mode='a', index=False, header=not os.path.exists(write_dir_sym+sym))

def to_30min(df):
    df = df.set_index('timestamp_exchange')
    df.index = pd.to_datetime(df.index)
    df['ret'] = np.log(df['close'] / df['open'])    
    df['above'] = df['close'] >= df['total_avg']
    df.index = pd.to_datetime(df.index)
    dfd = df.between_time('09:00', '15:00')
    dfn = df.between_time('21:00', '02:30')
    dfd = dfd.merge(dfd.groupby('symbol')['ret'].expanding().mean().reset_index().rename(columns={'ret': 'mean'}), on=['timestamp_exchange', 'symbol'], how='left').set_index('timestamp_exchange')
    dfd = dfd.merge(dfd.groupby('symbol')['ret'].expanding().std().reset_index().rename(columns={'ret': 'std'}), on=['timestamp_exchange', 'symbol'], how='left').set_index('timestamp_exchange')
    dfd['prev_above'] = dfd.groupby('symbol')['above'].shift(1)
    dfd['prev_above'] = dfd['prev_above'].fillna(dfd['above'])
    dfd['cross'] = (dfd['above'] != dfd['prev_above']).astype(int)
    bins = pd.IntervalIndex.from_arrays(ckpts[:-1], ckpts[1:], 'left')
    group = dfd.groupby(['symbol', pd.cut(dfd.index, bins)])
    dfd30 = group['open'].first().reset_index()
    dfd30['high'] = group['high'].max().reset_index()['high']
    dfd30['low'] = group['low'].min().reset_index()['low']
    dfd30['close'] = group['close'].last().reset_index()['close']
    dfd30['volume'] = group['volume'].sum().reset_index()['volume']
    dfd30['turnover'] = group['turnover'].sum().reset_index()['turnover']
    dfd30['opi'] = group['opi'].sum().reset_index()['opi']
    dfd30['contract'] = group['contract'].first().reset_index()['contract']
    dfd30['cmean'] = group['mean'].last().reset_index()['mean']
    dfd30['cstd'] = group['std'].last().reset_index()['std']
    dfd30['dmean'] = group['ret'].mean().reset_index()['ret']
    dfd30['dstd'] = group['ret'].std().reset_index()['ret']
    dfd30['cross'] = group['cross'].sum().reset_index()['cross']
    dfd30['date'] = date
    dfd30['timestamp_exchange'] = pd.to_datetime(dfd30['level_1'].apply(lambda x: x.right).astype(str))
    dfd30['time'] = dfd30['timestamp_exchange'].dt.strftime('%H:%M')
    dfd30['cum_cross'] = dfd30.groupby('symbol')['cross'].cumsum().reset_index()['cross']
    dfd30 = dfd30.drop(columns=['level_1'])
    if not dfn.empty:
        dfn = dfn.merge(dfn.groupby('symbol')['ret'].expanding().mean().reset_index().rename(columns={'ret': 'mean'}), on=['timestamp_exchange', 'symbol'], how='left').set_index('timestamp_exchange')
        dfn = dfn.merge(dfn.groupby('symbol')['ret'].expanding().std().reset_index().rename(columns={'ret': 'std'}), on=['timestamp_exchange', 'symbol'], how='left').set_index('timestamp_exchange')
        dfn['prev_above'] = dfn.groupby('symbol')['above'].shift(1)
        dfn['prev_above'] = dfn['prev_above'].fillna(dfn['above'])
        dfn['cross'] = (dfn['above'] != dfn['prev_above']).astype(int)
        group = dfn.groupby('symbol').resample('30min')
        dfn30 = group['open'].first().reset_index()
        dfn30['high'] = group['high'].max().reset_index()['high']
        dfn30['low'] = group['low'].min().reset_index()['low']
        dfn30['close'] = group['close'].last().reset_index()['close']
        dfn30['volume'] = group['volume'].sum().reset_index()['volume']
        dfn30['turnover'] = group['turnover'].sum().reset_index()['turnover']
        dfn30['opi'] = group['opi'].sum().reset_index()['opi']
        dfn30['contract'] = group['contract'].first().reset_index()['contract']
        dfn30['cmean'] = group['mean'].last().reset_index()['mean']
        dfn30['cstd'] = group['std'].last().reset_index()['std']
        dfn30['dmean'] = group['ret'].mean().reset_index()['ret']
        dfn30['dstd'] = group['ret'].std().reset_index()['ret']
        dfn30['cross'] = group['cross'].sum().reset_index()['cross']
        dfn30['date'] = date
        dfn30['timestamp_exchange'] = dfn30['timestamp_exchange'] + datetime.timedelta(minutes=30)
        dfn30['time'] = dfn30['timestamp_exchange'].dt.strftime('%H:%M')
        dfn30['cum_cross'] = dfn30.groupby('symbol')['cross'].cumsum().reset_index()['cross']
        dfn30 = dfn30[dfd30.columns]
        df30 = pd.concat([dfn30, dfd30])
    else:
        df30 = dfd30.copy()
    df30 = df30.sort_values(['symbol', 'timestamp_exchange']).reset_index(drop=True)
    df30['csharpe'] = df30['cmean'] / df30['cstd']
    df30['dsharpe'] = df30['dmean'] / df30['dstd']
    df30['adj_csharpe'] = df30['csharpe'] / (df30['cum_cross'] + 1)
    df30['adj_dsharpe'] = df30['dsharpe'] / (df30['cross'] + 1)
    df30 = df30.set_index('timestamp_exchange')
    # if ex == 'zce':
        # df30 = df30.drop(columns=['cross', 'cum_cross', 'adj_csharpe', 'adj_dsharpe'])
    df30 = df30.fillna(0.0)
    return df30

def add_rank(ex):
    df = pd.read_csv(write_dir_30min+ex+'.csv', delimiter='\t')
    df = df.groupby('symbol').tail(250).reset_index(drop=True)
    df['abs_csharpe'] = np.abs(df['csharpe'])
    df['abs_dsharpe'] = np.abs(df['dsharpe'])
    # if ex != 'zce':
    df['abs_adj_csharpe'] = np.abs(df['adj_csharpe'])
    df['abs_adj_dsharpe'] = np.abs(df['adj_dsharpe'])
    group = df.groupby('symbol')
    df['crank'] = group['abs_csharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_csharpe']]
    df['drank'] = group['abs_dsharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_dsharpe']]
    # if ex != 'zce':
    df['adj_crank'] = group['abs_adj_csharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_adj_csharpe']]
    df['adj_drank'] = group['abs_adj_dsharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_adj_dsharpe']]
    df = df.drop(columns=['cmean', 'dmean', 'cstd', 'dstd', 'abs_csharpe', 'abs_dsharpe'])
    # if ex != 'zce':
    df = df.drop(columns=['abs_adj_csharpe', 'abs_adj_dsharpe'])
    df = df[df['date'] == date]
    return df

def calculate_sig(df):
    night1 = -1
    night2 = -1
    day1 = -1
    day2 = -1
    dfn = df[(df['time']>='21:01') & (df['time']<='21:10')]
    dfd = df[(df['time']>='09:01') & (df['time']<='09:10')]
    if dfn.shape[0] > 1 and '21:01' in dfn['time'].unique():
        bm = dfn[dfn['time']=='21:01']['volume'].iloc[0]
        dfn['vol_flag'] = (dfn['volume']>bm).astype(int)
        night2 = dfn['vol_flag'].sum()
        night1 = dfn[dfn['time']<='21:05']['vol_flag'].sum()
    if '09:01' in dfd['time'].unique():
        bm = dfd[dfd['time']=='09:01']['volume'].iloc[0]
        dfd['vol_flag'] = (dfd['volume']>bm).astype(int)
        day2 = dfd['vol_flag'].sum()
        day1 = dfd[dfd['time']<='09:05']['vol_flag'].sum()
        idx = ['night_early_simple1', 'night_early_simple2', 'day_early_simple1', 'day_early_simple2']
    return pd.Series((night1, night2, day1, day2), index=idx)

def early_simple(df):
    dff = df[df['symbol'].str.slice(-1)!='1']
    group = dff.groupby(['date', 'symbol'])
    dfg = group.apply(calculate_sig).reset_index()
    dfg = dfg.sort_values(['date', 'symbol'])
    return dfg

if __name__ == '__main__':
    database_dir = '/shared/database_comdty/proto_'
    write_dir_exdate = '/shared/shared_data/comdty_data/1min_by_exdate/'
    write_dir_sym = '/shared/shared_data/comdty_data/1min_by_sym/'
    write_dir_30min = '/shared/shared_data/comdty_data/30min/'
    write_dir_sha = '/shared/strategy_stats/'
    write_dir_es = '/shared/strategy_stats/early_simple.csv'
    # write_dir_exdate = '/home/hyh/minute/'
    # write_dir_sym = '/home/hyh/minute/'
    # write_dir_30min = '/home/hyh/minute/30min/'
    # write_dir_sha = '/home/hyh/minute/30min/'
    work_dir = '/home/hyh/resample_data/'
    convert_dir = '/shared/shared_data/'
    sym_path = '/shared/database/Product_info.json'
    with open(sym_path,'r') as f:
        sym_info = json.loads(f.read())
    calendar = pd.read_csv('/shared/database/shfe_trading.calendar', delimiter='\t', index_col='date')
    date = int(datetime.datetime.today().date().strftime('%Y%m%d'))

    symbol_lists = {'zce': ['AP', 'CF', 'CJ', 'CY', 'FG', 'MA', 'OI', 'RM', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR', 'ZC', 'PF', 'PK'],
                    'dce': ['a', 'c', 'cs', 'eb', 'eg', 'i', 'j', 'jd', 'jm', 'l', 'lh', 'm', 'p', 'pg', 'pp', 'v', 'y'],
                    'shfe': ['ag', 'al', 'au', 'bu', 'cu', 'fu', 'hc', 'ni', 'pb', 'rb', 'ru', 'sn', 'sp', 'ss', 'zn', 'lu', 'sc']}

    useful_fields = ['instr_id', 'trading_day', 'action_day', 'local_ts', 'time', 'second', 'milli_sec', 'lastprice', 'tick_volume', 'tick_turnover', 'total_volume', 'total_turnover', 'total_average', 'total_opi', 'pre_opi', 'pre_close']
    eff_cols = ['timestamp_exchange', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'opi', 'symbol', 'contract', 'date', 'time', 'total_avg']

    log_file = work_dir+str(date)+"error.log"
    logging.basicConfig(filename=log_file, filemode='a', format='%(levelname)s %(asctime)s - %(message)s', level=logging.ERROR)
    logger = logging.getLogger()

    checkpoints = ['09:00:00', '09:30:00', '10:00:00', '10:45:00','11:15:00','13:45:00','14:15:00','15:00:00']
    ckpts = pd.to_datetime(list(map(lambda x: str(date)+' '+x, checkpoints)))

    for ex in ['dce', 'shfe']:
        dfc = pd.read_csv('/shared/shared_data/convert_jq.'+ex+'.day.csv', delimiter='\t')
        dfcon = pd.read_csv('/shared/shared_data/convert_data.'+ex+'.day.csv', delimiter='\t')
        output_path = write_dir_exdate + ex + '_' + str(date)
        try:
            df = to_minute(ex)
        except:
            logger.error('Something wrong when converting '+str(date)+' '+ex+' to minute data')
            continue
        if df.empty:
            logger.error('Something wrong when converting '+str(date)+' '+ex+' to minute data')
            continue
        df.to_csv(output_path, sep='\t', mode='a', index=False, header=not os.path.exists(output_path))
        to_sym(df)
        try:
            df30 = to_30min(df)
            df30.to_csv(write_dir_30min+ex+'.csv', sep='\t', mode='a', header=False)
        except:
            logger.error('Something wrong when converting '+str(date)+' '+ex+' to 30-min data')
        try:
            dfsha = add_rank(ex)
            dfsha.to_csv(write_dir_sha+ex+'_sha.csv', sep='\t', mode='a', header=False, index=False)
        except:
            logger.error('Something wrong when adding sharpe rank to '+str(date)+' '+ex+' data')
        try:
            dfes = early_simple(df)
        except:
            logger.error('Something wrong when converting '+str(date)+' '+ex+' to early simple data')
            continue
        dfes.to_csv(write_dir_es, sep='\t', mode='a', header=False, index=False)
    

    if os.path.exists(log_file) and os.stat(log_file).st_size == 0:
        os.remove(log_file)
