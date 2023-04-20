# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os
import time
import datetime
from lt_commodity_settings import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pd.set_option('display.float_format',lambda x: '%.4f' % x)
pd.set_option('display.max_colwidth',200)
pd.set_option('display.max_rows',50000)
pd.set_option('display.max_columns',40)
pd.set_option('expand_frame_repr', False)

cols=['date','time','open','high','low','close','volume']

MA_NUMBER=60
NUMBER_SMALL = 5
NUMBER_BIG = 20

list_exch=['dce','zce','shfe','cffex']

def get_stoptarget(symbol):
   data = pd.read_csv('/shared/strategy_stats/volatility_stats_30mins.csv',sep='\t')
   data['target'] = data['100mean']+2.5*data['100std']
   data1 = data[data.date == data['date'].max()].set_index('symbol')
   print(data1)
   try:
      target = data1.loc[symbol]['target']
   except:
      target = 3
   # print(target)
   return target

def get_convert_data():
   df_convert=pd.DataFrame()
   for exch in list_exch:
      convert_day_file='/shared/shared_data/convert_data.'+exch+'.day.csv'
      convert_night_file='/shared/shared_data/convert_data.'+exch+'.night.csv'
      df_convert_day=pd.read_csv(convert_day_file, sep='\t')
      if exch != 'cffex':
         df_convert_night=pd.read_csv(convert_night_file, sep='\t')
         df_convert_exch=pd.concat([df_convert_day, df_convert_night]).reset_index(drop=True)
      else:
         df_convert_exch = df_convert_day.copy()
      df_convert=pd.concat([df_convert, df_convert_exch]).reset_index(drop=True)
      df_convert['time'] = pd.to_datetime(df_convert['time'])
      df_convert = df_convert.sort_values('time')
   return df_convert

"""
def get_convert_data():
   df = pd.read_csv('/shared/shared_data/comdty_data/30min/dce.csv', delimiter='\t')
   df1 = pd.read_csv('/shared/shared_data/comdty_data/30min/shfe.csv', delimiter='\t') 
   df2 = pd.read_csv('/shared/shared_data/comdty_data/30min/zce.csv', delimiter='\t')
   df = pd.concat([df, df1, df2])
   df['time'] = df['timestamp_exchange']
   df['time_hour'] = df['time'].str.slice(-8, )   
   return df
"""

def get_sharpe_data():
   df_sha = []
   for exch in list_exch:
      sha = pd.read_csv('/shared/strategy_stats/%s_sha.csv'%exch, delimiter='\t')
      sha = sha[sha['symbol'].str.slice(-1) != '1']
      df_sha.append(sha)
   df_sha = pd.concat(df_sha)
   return df_sha

def generate_strategy_info_lt_commodity(date, df_convert, df_sharpe, session):
   strategy_info={}
   for symbol in list_symbol:
      strategy=symbol+'_all_sig'
      df_convert_symbol=df_convert[df_convert['symbol']==symbol]
      df_sharpe_symbol=df_sha[df_sha['symbol']==symbol]
      model_info=get_strategy_info_longterm(strategy, symbol, df_convert_symbol, df_sharpe_symbol, date, session)
      if model_info is not None:
         strategy_info[strategy]=model_info.copy()
   return strategy_info

def get_live(symbol):
   if symbol in list_exception:
      live = 0
   else:
      live = 1
   return live

def get_multiplier(symbol):
   with open('/shared/database/Product_info.json', 'r') as fp:
      product_info = json.load(fp)
   
   print(symbol, product_info[symbol]['multiple'])
   return product_info[symbol]['multiple']

def get_wait_time(symbol):
   # data = pd.read_csv('/home/hyh/longterm_time_to_market_order/longterm_to_market_time.csv', sep='\t',index_col='symbol')
   data = pd.read_csv('/home/hyh/longterm_time_to_market_order/120w.csv', sep='\t',index_col='symbol')
   try:
      wait_time = data['wait_time'].loc[symbol]
   except:
      wait_time = 30
   return wait_time

def get_re_thresh(date, session, symbol):
   if session == 'night':
      get_date = int(date)
   elif session == 'day':
      try:
         get_date = int(calendar.loc[int(date), 'prev_date'])
      except KeyError:
         get_date = int(calendar.index[-1])
   df = pd.read_csv('/shared/strategy_stats/latest_thresh_lamp_price.csv', delimiter='\t', index_col=0)
   dfs = df.loc[symbol]
   if np.isnan(dfs['reverse_from_high']) or dfs['reverse_from_high'] == 50 or np.isnan(dfs['high/sett']) or np.isnan(dfs['low/sett']):
      flag = 0
      high = 0.5
      low = 0.5
   else:
      flag = 1
      high = dfs['reverse_from_high'] / 100
      low = dfs['reverse_from_low'] / 100
   return dfs['high/sett'], dfs['low/sett'], flag, high, low
      
def get_prev_sharpe(symbol, df, date, session, zce):
   if session == 'day':
      if date == last_date:
         return sha_his[symbol]
      elif zce:
         return df[df['date'] <= date]['csharpe'].iloc[-250:-7].tolist()
      else:
         return df[df['date'] <= date]['adj_csharpe'].iloc[-250:-7].tolist()
   else:
      if zce:
         return df[df['date'] <= date]['csharpe'].iloc[-250:].tolist()
      else:
         return df[df['date'] <= date]['adj_csharpe'].iloc[-250:].tolist()


def get_strategy_info_longterm(strategy, symbol, df_convert_symbol, df_sha_symbol, date, session):
   model_info={}
   model_info['path']=['/shared/zce/day/20220802']
   model_info['backtrade']=int(ARGS.backtrade)
   model_info['broker']= broker
   # model_info['live']=get_live(symbol)
   model_info['live']=0
   model_info['session']=session
   model_info['flatten']=0
   model_info['wait_time'] = int(get_wait_time(symbol))
# model_info['day_ts'] = ['09:29:30', '09:59:30', '10:44:30', '11:14:30', '13:44:30', '14:14:30', '14:54:30', '15:27:30']
#model_info['night_ts'] = ['21:29:30', '21:59:30', '22:29:30', '22:57:30', '23:27:30', '23:57:30', '00:27:30', '00:57:30', '01:27:30', '01:57:30', '02:27:30', '02:57:30']
   model_info['day_ts'] = ['09:30', '10:00', '10:45', '11:15', '13:45', '14:15', '14:55', '15:28']
   #model_info['day_ts'] = ['13:45', '14:15', '14:55', '15:28']
   model_info['night_ts'] = ['21:30', '22:00', '22:30', '22:58', '23:28', '23:58', '00:28', '00:58', '01:28', '01:58', '02:28', '02:58']
#  model_info['twap_total_time'] = 30
#  model_info['twap_gap_time'] = 100
   contract=get_contract(date, symbol, df_convert_symbol)
   print('contract=',contract)
   if contract is None:
      print('error: find contract for '+symbol+' on date '+str(date))
      return None
   model_info['symbol']=symbol
   model_info['contract']=[contract]
   model_info['multiplier']=get_multiplier(symbol)
   model_info['end_check']=get_end_check(symbol)
   model_info['stop_loss_chg']=min(3.0, get_stoptarget(symbol))/100.0
   model_info['take_profit_chg']=4*model_info['stop_loss_chg']
   # model_info['stop_loss_chg']=0.002
   # model_info['take_profit_chg']=0.002
   model_info['market_value']=market_value
   # model_info['open_long'], model_info['flat_long'], model_info['open_short'], model_info['flat_short']=get_thresh_info(symbol)
   model_info['prev_vol'], model_info['prev_open'], model_info['prev_high'], model_info['prev_low'], model_info['prev_close'], model_info['last_time']=get_prev_info(date, session, df_convert_symbol, MA_NUMBER)
   #model_info['cover'] = 1
   #model_info['cover_multiplier'] = 0.7
   model_info['twap'] = 0
   # if symbol in ['TA', 'MA', 'eg', 'PF', 'bu', 'rb', 'm', 'FG']:
   # model_info['twap'] = 1 
   # if session == 'night':
   #    model_info['session_end'] = get_end_check(symbol)[:-1] + '8'
   #    if symbol not in ['sc','ag','au', 'cu','al','zn','pb','ni','sn','ss']:
   #       model_info['session_end_date'] = str(date)
   #    else:
   #       model_info['session_end_date'] = (datetime.datetime.strptime(str(date), '%Y%m%d')+datetime.timedelta(days=1)).strftime('%Y%m%d')
   # else:
   #    model_info['session_end'] = '14:58'
   #    model_info['session_end_date'] = str(date) 
   if symbol in symbol_lists['zce']:
      model_info['is_zce'] = 1
   else:
      model_info['is_zce'] = 0
   # elif symbol in symbol_lists['dce']:
   #    model_info['flatten_method'] = '1dip'
   # elif symbol in symbol_lists['shfe']:
   #    model_info['flatten_method'] = '2dip'
   # model_info['em_stop_loss_chg'] = 0.005
   # model_info['high_thresh'], model_info['low_thresh'], model_info['re_stop_loss_flag'], model_info['re_stop_loss_chg_high'], model_info['re_stop_loss_chg_low'] = get_re_thresh(date, session, symbol)
   model_info['prev_csharpe'] = get_prev_sharpe(symbol, df_sha_symbol, int(date), session, model_info['is_zce'])
   return model_info

def get_end_check(symbol):
   if symbol in ['sc','ag','au']:
      return '02:28'
   elif symbol in ['cu','al','zn','pb','ni','sn','ss']:
      return '00:58'
   else:
      return '22:58'

def get_contract(date, symbol, df_convert):
   df_convert_date=df_convert[df_convert['date']==int(date)]
   df_convert_date=df_convert_date.sort_values(by='time', ascending=True)
   if len(df_convert_date)<1:
      return df_convert['contract'].iloc[-1]
   else:
      return df_convert_date['contract'].iloc[0]
   # return df_convert['contract'].iloc[-1]

def get_thresh_info(symbol):
   if symbol in list_long:
      return 3, -4, -5, 2
   elif symbol in list_short:
      return 5, -2, -3, 4
   else:
      return 4, -3, -4, 3

def get_prev_info(date, session, df_convert_symbol, MA_NUMBER):
   date_str = str(date)[:4] + '-' + str(date)[4:6] + '-' + str(date)[6:]
   if session == 'day':
      df_symbol=df_convert_symbol[df_convert_symbol['date']<int(date)]
   elif session == 'night':
      df_symbol=df_convert_symbol[df_convert_symbol['time']<date_str+' 21:00:00']
   df_symbol=df_symbol.sort_values(by='time', ascending=True)
   df_symbol=df_symbol[df_symbol['volume']>0]
   df_symbol=df_symbol.tail(MA_NUMBER)
   print(df_symbol[cols].tail(10))
   list_vol=df_symbol['volume'].tolist()
   list_open=df_symbol['open'].tolist()
   list_high=df_symbol['high'].tolist()
   list_low=df_symbol['low'].tolist()
   list_close=df_symbol['close'].tolist()
   last_time=df_symbol['time'].astype(str).iloc[-1]
   return list_vol, list_open, list_high, list_low, list_close, last_time

if __name__=="__main__":
   # cal default last_date
   data = get_convert_data()
   last_date = int(time.strftime('%Y%m%d'))

   # cal session
   if int(time.strftime('%H'))>=16:
      today_session = 'night'
   else:
      today_session = 'day'

   with open('/shared/aqs_trade_conf/server_info.json') as f:
      brokers =  json.load(f)['SJ']['account']

   PARSER = argparse.ArgumentParser()
   # PARSER.add_argument('-sim','--sim',action='store_true')
   PARSER.add_argument('-date','--date',help='date', default=last_date)
   PARSER.add_argument('-session','--session',help='session', default = today_session)
   PARSER.add_argument('-broker',help='brokers:', default = None)
   PARSER.add_argument('-user',help='strategy runner', default = None)
   PARSER.add_argument('-ex',help='exchange', default = 'all')
   PARSER.add_argument('-symbol',help='symbols', nargs='+', default = [])
   PARSER.add_argument('-backtrade', action='store_true')
   ARGS = PARSER.parse_args()

   if ARGS.broker == None:
      broker = brokers
   else:
      broker = ARGS.broker.split(',')

   if ARGS.user == None:
      user = 'longterm'
   else:
      user = ARGS.user

   if ARGS.ex != 'all':
      list_symbol = symbol_lists[ARGS.ex]

   if ARGS.symbol != []:
      list_symbol = ARGS.symbol

   calendar = pd.read_csv('/shared/database/shfe_trading.calendar', delimiter='\t', index_col=0)

   df_convert = get_convert_data()
   df_sha = get_sharpe_data()
   with open('/home/hyh/intraday_sharpe/sharpe_history.json', 'r') as f:
      sha_his = json.load(f)
   df_convert = df_convert.drop_duplicates(subset=['time','symbol'])
   strategy_info = generate_strategy_info_lt_commodity(ARGS.date, df_convert, df_sha, ARGS.session)
   if strategy_info is None:
      print('failed generating strategy info')
      sys.exit(0)
   
   strategy_exe_file = '/home/public/monitor/strategy_conf.json'
   with open(strategy_exe_file, 'w') as f:
        json.dump(strategy_info, f, indent=4)
