import jqdatasdk as jq
import pandas as pd
import datetime
import os
import json
import logging
import numpy as np

jq.auth('17621627675', 'Zytz2020')

def get_contract(x, date, dfc, dfcon):
    a = dfcon[(dfcon['date']==date)&(dfcon['symbol']==x)]['contract']
    if not a.empty:
        return a.iloc[0]
    # b = dfc[(dfc['date']==date)&(dfc['symbol']==x)]['contract']
    # if not b.empty:
    #     return b.iloc[0]
    try:
        if x[-1] != '1':
            return dfmain.loc[date, x]['dominant']
        else:
            if x == 'ZC1':
                return 'ZC307'
            return dfmain.loc[date, x[:-1]]['sub_domi'] 
    except:
        b = dfc[(dfc['date']==date)&(dfc['symbol']==x)]['contract']
        if not b.empty:
            return b.iloc[0]


def download_jq_data(s, con, ex, logger):
    if s in ['sc', 'nr', 'bc', 'lu', 'sc1', 'nr1', 'bc1', 'lu1']:
        downcon = con.upper()+'.XINE'
    else:
        downcon = con.upper()+'.'+dct[ex]
    if ex == 'zce':
        if downcon[2] <= '5':
            downcon = downcon[:2] + '2' + downcon[2:]
        else:
            downcon = downcon[:2] + '1' + downcon[2:]
    prev = int(calendar.loc[date, 'prev_date'])
    sdate = datetime.datetime.strptime(str(prev), '%Y%m%d').strftime('%Y-%m-%d')
    edate = (datetime.datetime.strptime(str(date), '%Y%m%d')+datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    data = jq.get_price(downcon, start_date=sdate, end_date=edate, frequency='minute', fields=['open', 'high', 'low', 'close', 'volume', 'money', 'open_interest'])
    if data.empty:
        logger.error('%s, %s, %d, no data' % (sym, con, date))
        return data
    data['symbol'] = s
    data['contract'] = con
    return data

def change_date(row):
    if row['time'][:2] >= '21' and row['time'][:2] <= '23':
        return int(calendar.loc[row['date'], 'next_date'])
    elif row['time'][:2] >= '00' and row['time'][:2] <= '02':
        return int(calendar.loc[int((datetime.datetime.strptime(str(row['date']), '%Y%m%d')-datetime.timedelta(days=1)).strftime('%Y%m%d')), 'next_date'])
    else:
        return row['date']

def convert_jq(data, date, sym, con, logger):
    data.index = pd.to_datetime(data.index)
    date_str = str(date)[:4]+'-'+str(date)[4:6]+'-'+str(date)[-2:]+' '+'16:00:00'
    data = data.loc[:date_str]
    data = data.reset_index()
    data['index'] = pd.to_datetime(data['index'])
    data['index'] = data['index'] - pd.Timedelta('1m')
    data['date'] = data['index'].dt.strftime('%Y%m%d').astype(int)
    data['time'] = data['index'].dt.strftime('%H:%M')
    data['date'] = data.apply(change_date, axis=1)
    data['pre_close'] = data['close'].shift(1)
    df = data[(data['symbol']==sym) & (data['date']==date)]
    df = df.drop_duplicates(['symbol', 'time', 'date'])
    df = df.rename(columns={'money': 'turnover', 'index': 'timestamp_exchange'})
    prev = int(calendar.loc[date, 'prev_date'])
    contract = df['contract'].iloc[0]
    df['opi'] = df['open_interest'].diff(1)
    try:
        prev_opi = data[(data['contract']==contract) & (data['date']==prev)]['open_interest'].iloc[-1]
        prev_close = data[(data['contract']==contract) & (data['date']==prev)]['close'].iloc[-1]
    except:
        prev_opi = 0
        prev_close = -1
        logger.error('%d, %s, %s, can\' get data from last session' % (date, sym, con))
    # try:
    #     df['opi'].iloc[0] = df['open_interest'].iloc[0] - data[(data['contract']==contract) & (data['date']==prev)]['open_interest'].iloc[-1]
    # except:
    #     df['opi'].iloc[0] = df['open_interest'].iloc[0]
    #     logger.error('%d, %s, %s, opi calculation error' % (date, sym, con))
    df['opi'].iloc[0] = df['open_interest'].iloc[0] - prev_opi
    df['money_flow_wh'] = df['open_interest'] * df['close'] - prev_opi * prev_close
    df['total_avg'] = df['turnover'].cumsum() / df['volume'].cumsum() / mult[sym]
    df = df[['timestamp_exchange', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'opi', 'symbol', 'contract', 'date', 'time', 'total_avg', 'pre_close', 'money_flow_wh']]
    return df

def to_sym(df):
    for sym in df['symbol'].unique():
        dd = pd.read_csv(write_dir_sym+sym, delimiter='\t', usecols=['date'])
        if dd['date'].max() == date:
            dd = pd.read_csv(write_dir_sym+sym, delimiter='\t')
            dd = dd[dd['date'] < date]
            dd.to_csv(write_dir_sym+sym, sep='\t', index=False)
        dfc = df[df['symbol'] == sym]
        dfc.to_csv(write_dir_sym+sym, sep='\t', mode='a', index=False, header=False)


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

def money_flow_statistics(df):
    df = df.reset_index(drop=True)
    df['money_flow_chg'] = df.groupby('symbol')['money_flow_wh'].diff()
    group = df.groupby('symbol')
    a = group['date'].first().to_frame()
    a['std'] = group['money_flow_wh'].std()
    a['99_percentile'] = group['money_flow_wh'].quantile(0.99)
    a['95_percentile'] = group['money_flow_wh'].quantile(0.95)
    a['5_percentile'] = group['money_flow_wh'].quantile(0.05)
    a['1_percentile'] = group['money_flow_wh'].quantile(0.01)
    a['mf_change_mean'] = group['money_flow_chg'].mean()
    a['mf_change_std'] = group['money_flow_chg'].std()
    a = a.reset_index().set_index('date').reset_index()
    return a

if __name__ == '__main__':
    symbol_lists = {'zce': ['AP', 'CF', 'CJ', 'CY', 'FG', 'MA', 'OI', 'RM', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR', 'ZC', 'PF', 'PK'],
                    'dce': ['a', 'c', 'cs', 'eb', 'eg', 'i', 'j', 'jd', 'jm', 'l', 'lh', 'm', 'p', 'pg', 'pp', 'v', 'y'],
                    'shfe': ['ag', 'al', 'au', 'bu', 'cu', 'fu', 'hc', 'ni', 'pb', 'rb', 'ru', 'sn', 'sp', 'ss', 'zn', 'lu', 'sc']}

    dct = {'zce': 'XZCE', 'dce': 'XDCE', 'shfe': 'XSGE'}

    date = int(datetime.datetime.today().strftime('%Y%m%d'))

    dfmain = pd.read_csv('/shared/shared_data/comdty_data/daily/dominant_contract.csv', delimiter='\t', index_col=0, header=[0,1])
    calendar = pd.read_csv('/shared/database/shfe_trading.calendar', delimiter='\t', index_col=0)
    calendar['next_date'].iloc[-1] = date
    work_dir = '/home/hyh/resample_data/'
    # write_dir = '/home/hyh/minute/exdate/'
    # write_dir_sym = '/home/hyh/minute/sym/'
    write_dir = '/shared/shared_data/comdty_data/1min_by_exdate/'
    write_dir_sym = '/shared/shared_data/comdty_data/1min_by_sym/'
    write_dir_30min = '/shared/shared_data/comdty_data/30min/'
    write_dir_sha = '/shared/strategy_stats/'
    write_dir_es = '/shared/strategy_stats/early_simple.csv'
    write_dir_mf = '/shared/strategy_stats/money_flow_statistics.csv'
    with open('/shared/database/Product_info.json','r') as f:
        sym_info = json.loads(f.read())
    mult = {}
    for s in symbol_lists['dce']+symbol_lists['shfe']+symbol_lists['zce']:
        mult[s] = sym_info[s]['multiple']
        mult[s+'1'] = sym_info[s]['multiple']

    log_file = work_dir+str(date)+"error.log"
    logging.basicConfig(filename=log_file, filemode='a', format='%(levelname)s %(asctime)s - %(message)s', level=logging.ERROR)
    logger = logging.getLogger()

    checkpoints = ['09:00:00', '09:30:00', '10:00:00', '10:45:00','11:15:00','13:45:00','14:15:00','15:00:00']
    ckpts = pd.to_datetime(list(map(lambda x: str(date)+' '+x, checkpoints)))

    for ex in ['shfe', 'dce', 'zce']:
        syms = symbol_lists[ex]
        syms += list(map(lambda x: x+'1', syms))
        dfc = pd.read_csv('/shared/shared_data/convert_jq.'+ex+'.day.csv', delimiter='\t')
        dfcon = pd.read_csv('/shared/shared_data/convert_data.'+ex+'.day.csv', delimiter='\t')
        ret = []
        for sym in syms:
            con = get_contract(sym, date, dfc, dfcon)
            data = download_jq_data(sym, con, ex, logger)
            if data.empty:
                logger.error('no data for %s, %d' % (sym, date))
                continue
            try:
                dfmin = convert_jq(data, date, sym, con, logger)
            except:
                logger.error('something wrong with %s, %d when converting to minute data' % (sym, date))
                continue
            ret.append(dfmin)
        df = pd.concat(ret)
        df = df.sort_values(['symbol', 'timestamp_exchange'])
        df.to_csv(write_dir+ex+'_'+str(date), sep='\t', index=False)
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
        try:
            dfmf = money_flow_statistics(df)
        except:
            logger.error('Something wrong when converting '+str(date)+' '+ex+' to money flow statistics')
            continue
        dfmf.to_csv(write_dir_mf, sep='\t', mode='a', header=False, index=False)
    
    if os.path.exists(log_file) and os.stat(log_file).st_size == 0:
        os.remove(log_file)
        

    

