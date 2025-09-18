import datetime
tmp=(datetime.date.today() - datetime.timedelta(1))
end_date = tmp.strftime('%Y-%m-%d')
start_date  = (tmp - datetime.timedelta(28)).strftime('%Y-%m-%d')
