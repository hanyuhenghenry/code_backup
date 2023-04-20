# broker=['msb']
market_value=50000

list_symbol=['i','j','jm','rb','hc','SF','SM','ss','SA','FG',\
             'sc','fu','bu','ru','l','pp','v','eg','eb','pg','PF','TA','MA', \
             'cu','al','zn','pb','ni','sn',\
             'ag','au',\
             'SR','CF','RM','OI','AP','m','a','y','p','c',\
             'UR','CJ','AP','sp','jd','lh']
list_long=[]
list_neutral=['i','j','jm','rb','hc','ZC','SF','SM','ss','SA','FG',\
             'sc','fu','bu','ru','l','pp','v','eg','eb','pg','PF','TA','MA','lu',\
             'cu','al','zn','pb','ni','sn',\
             'ag','au',\
             'SR','CF','RM','OI','AP','m','a','y','p','c',\
             'UR','CJ','AP','sp','jd','lh',\
             'IC','IH','IF','T','TF','TS']
list_short=[]

list_exception = ['i','j','jm','rb','hc','ZC','SF','SM','ss','SA','FG',\
             'sc','fu','bu','ru','l','pp','v','eg','eb','pg','PF','TA','MA','lu',\
             'cu','al','zn','pb','ni','sn',\
             'ag','au',\
             'SR','CF','RM','OI','AP','m','a','y','p','c',\
             'UR','CJ','AP','sp','jd','lh',\
             'IC','IH','IF','T','TF','TS']

list_items = ['market_value','symbol','prev_open','prev_close',\
    'broker','session','flatten','multiplier','open_long','path','open_short',\
    'end_check','backtrade','prev_high','prev_low','contract','last_time',\
    'wait_time','live','flat_long','flatten_px_chg','prev_vol','flat_short']

symbol_lists = {'zce': ['AP', 'CF', 'CJ', 'FG', 'MA', 'OI', 'RM', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR', 'ZC', 'PF', 'PK'],
                    'dce': ['a', 'c', 'cs', 'eb', 'eg', 'i', 'j', 'jd', 'jm', 'l', 'lh', 'm', 'p', 'pg', 'pp', 'v', 'y'],
                    'shfe': ['ag', 'al', 'au', 'bu', 'cu', 'fu', 'hc', 'ni', 'pb', 'rb', 'ru', 'sn', 'sp', 'ss', 'zn', 'lu', 'sc']}
