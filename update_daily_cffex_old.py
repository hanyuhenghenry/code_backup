import glob
import os
import pandas as pd
import numpy as np
import json
import re
import datetime
import sys
import logging

    

def to_minute():
    dfret = pd.DataFrame()
    output_path = work_dir + str(date)
    df = pd.read_hdf(database_dir+str(date)+'.h5')
    for sym in syms:
        contract = dfc.loc[(date, sym), 'contract'][0]
        dft = clean_tick_data_for_min(df, date, contract)
        if dft.empty:
            continue
        res = convert_to_min(dft, date, sym)
        dfret = pd.concat([dfret, res])
    return dfret


def convert_to_min(df, date, sym):
    dfm = df['lastprice'].resample('1min').ohlc()
    dfm['volume'] = df['volume'].resample('1min').max()
    dfm['turnover'] = df['turnover'].resample('1min').max()
    dfm = dfm.between_time('13:00', '11:29')
    dfm[['close', 'volume', 'turnover']] = dfm[['close', 'volume', 'turnover']].fillna(method='ffill')
    for col in ['open', 'high', 'low']:
        dfm[col] = dfm[col].fillna(dfm['close'])
    dfm['volume'][1:] = dfm['volume'].diff(1)[1:]
    dfm['turnover'][1:] = dfm['turnover'].diff(1)[1:]
    dfm['symbol'] = sym
    dfm['contract'] = df['symbol'][0]
    dfm = dfm.reset_index()
    dfm['date'] = date
    dfm['time'] = dfm['timestamp_exchange'].dt.strftime('%H:%M')
    dfm = dfm.merge(mult, how='left', on='symbol')
    dfm['total_avg'] = dfm.groupby('symbol')['turnover'].cumsum() / dfm.groupby('symbol')['volume'].cumsum() / dfm['mult']
    dfm = dfm.drop(columns=['mult'])
    return dfm


def change_index(df, newtime):
    df = df.reset_index()
    timestamp = df.iloc[0, 0]
    df.iloc[0, 0] = pd.to_datetime(str(timestamp).split()[0]+' '+newtime)
    df = df.set_index('timestamp_exchange')
    return df


def clean_tick_data_for_min(df, date, contract):
    sym = re.sub('[^a-zA-Z]+', '', contract)
    eff_cols = ['askprice1', 'askvolume1', 'bidprice1', 'bidvolume1', 'lastprice', 'symbol', 'openinterest', 'timestamp_exchange', 'turnover', 'volume']
    df = df[eff_cols]
    df = df[df['symbol']==contract]
    if df.empty:
        return df
    df = df.sort_values(['timestamp_exchange', 'volume'])
    df = df.set_index('timestamp_exchange') 
    if sym[0] == 'I':
        before = df.between_time('09:29', '09:30', include_end=False)
        first = df.between_time('09:30', '11:30', include_end = False)
        break1 = df.between_time('11:30', '11:35', include_end=False)
        second = df.between_time('13:00', '15:00', include_end=False)
        after = df.between_time('15:00', '15:01')
        firsttick = pd.DataFrame()
        break1tick = pd.DataFrame()
        lasttick = pd.DataFrame()
        if not before.empty: 
            firsttick = before.iloc[[-1]]
        elif not first.empty:
            firsttick = first.iloc[[0]]
        if not firsttick.empty:
            firsttick = change_index(firsttick, '09:30:00.000')
        if not break1.empty:
            if not all(first.iloc[-1]==break1.iloc[0]):
                break1tick = break1.iloc[[0]]
                break1tick = change_index(break1tick, '11:29:59.999')
        if not after.empty:
            lasttick = after.iloc[[0]]
        elif not second.empty:
            lasttick = second.iloc[[-1]]
        else:
            hours = pd.concat([first, break1tick, second])
            if hours.empty:
                return hours
            lasttick = hours.iloc[[-1]]
        lasttick = change_index(lasttick, '14:59:59.999')
        ret = pd.concat([firsttick, first, break1tick, second, lasttick])
        ret['vdiff'] = ret['volume'].diff(1)
        ret.ix[-1, 'vdiff'] = 1.0
        ret = ret[ret['vdiff'] != 0]
        cut1 = pd.to_datetime(str(date)+' 09:30')
        cut2 = pd.to_datetime(str(date)+' 15:00')
        ret = ret[(ret.index >= cut1) & (ret.index <= cut2)]
        return ret
    elif sym[0] == 'T':
        if date >= 20200720:
            before = df.between_time('09:29', '09:30', include_end=False)
            first = df.between_time('09:30', '11:30', include_end = False)
            start_time = '09:30'
        else:
            before = df.between_time('09:14', '09:15', include_end=False)
            first = df.between_time('09:15', '11:30', include_end = False)
            start_time = '09:15'
        break1 = df.between_time('11:30', '11:35', include_end=False)
        second = df.between_time('13:00', '15:15', include_end=False)
        after = df.between_time('15:15', '15:16')
        firsttick = pd.DataFrame()
        break1tick = pd.DataFrame()
        lasttick = pd.DataFrame()
        if not before.empty: 
            firsttick = before.iloc[[-1]]
        elif not first.empty:
            firsttick = first.iloc[[0]]
        if not firsttick.empty:
            firsttick = change_index(firsttick, start_time+':00.000')
        if not break1.empty:
            if not all(first.iloc[-1]==break1.iloc[0]):
                break1tick = break1.iloc[[0]]
                break1tick = change_index(break1tick, '11:29:59.999')
        if not after.empty:
            lasttick = after.iloc[[0]]
        elif not second.empty:
            lasttick = second.iloc[[-1]]
        else:
            hours = pd.concat([first, break1tick, second])
            if hours.empty:
                return hours
            lasttick = hours.iloc[[-1]]
        lasttick = change_index(lasttick, '15:14:59.999')
        ret = pd.concat([firsttick, first, break1tick, second, lasttick])
        ret['vdiff'] = ret['volume'].diff(1)
        ret.ix[-1, 'vdiff'] = 1.0
        ret = ret[ret['vdiff'] != 0]
        cut1 = pd.to_datetime(str(date)+' '+start_time)
        cut2 = pd.to_datetime(str(date)+' 15:15')
        ret = ret[(ret.index >= cut1) & (ret.index <= cut2)]
        return ret

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
        bins = pd.IntervalIndex.from_arrays(ckpts[:-1], ckpts[1:], 'left')
        group = dfd.groupby(['symbol', pd.cut(dfd.index, bins)])
        dfd30 = group['open'].first().reset_index()
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
    if ex != 'zce':
        df['abs_adj_csharpe'] = np.abs(df['adj_csharpe'])
        df['abs_adj_dsharpe'] = np.abs(df['adj_dsharpe'])
    group = df.groupby('symbol')
    df['crank'] = group['abs_csharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_csharpe']]
    df['drank'] = group['abs_dsharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_dsharpe']]
    if ex != 'zce':
        df['adj_crank'] = group['abs_adj_csharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_adj_csharpe']]
        df['adj_drank'] = group['abs_adj_dsharpe'].rolling(200, 200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).reset_index(0).sort_index()[['abs_adj_dsharpe']]
    df = df.drop(columns=['cmean', 'dmean', 'cstd', 'dstd', 'abs_csharpe', 'abs_dsharpe'])
    if ex != 'zce':
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
    dfc = pd.read_csv(convert_dir+'convert_data.cffex.day.csv', delimiter='\t', index_col=['date', 'symbol'])
    calendar = pd.read_csv('/shared/database/shfe_trading.calendar', delimiter='\t', index_col='date')
    date = int(datetime.datetime.today().date().strftime('%Y%m%d'))
    syms = ['IF', 'IC', 'IH', 'T', 'TS', 'TF', 'IF1', 'IC1', 'IH1', 'T1', 'TS1', 'TF1', 'IM', 'IM1']

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
    logging.basicConfig(filename=log_file, filemode='a', format='%(levelname)s %(asctime)s - %(message)s', level=logging.ERROR)
    logger = logging.getLogger()

    output_path = write_dir_date + str(date)
    # try:
    df = to_minute()
    # except:
    #     logger.error('Something wrong when converting '+str(date)+' to minute data')
    #     sys.exit()
    df.to_csv(output_path, sep='\t', mode='a', index=False, header=not os.path.exists(output_path))
    to_sym(df)
    try:
        df30 = to_30min(df)
    except:
        logger.error('Something wrong when converting '+str(date)+' to 30 minute data')
        sys.exit()
    df30.to_csv(write_dir_30min+'cffex.csv', sep='\t', mode='a', header=False)
    try:
        dfsha = add_rank()
    except:
        logger.error('Something wrong when adding sharpe rank to '+str(date)+' data')
        sys.exit()
    dfsha.to_csv(write_dir_sha+'cffex_sha.csv', sep='\t', mode='a', header=False, index=False)
    try:
        dfes = early_simple(df)
    except:
        logger.error('Something wrong when converting '+str(date)+' to early simple data')
        sys.exit()
    dfes.to_csv(write_dir_es, sep='\t', mode='a', header=False, index=False)

    if os.path.exists(log_file) and os.stat(log_file).st_size == 0:
        os.remove(log_file)
