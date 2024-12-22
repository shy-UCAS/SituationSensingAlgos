from datetime import datetime
from problog.extern import problog_export, problog_export_nondet

@problog_export('+str', '+str', '-int')
def compare_dates_int(date1, date2):
    _date_format = "%Y-%m-%d %H:%M:%S.%f"
    
    _d1 = datetime.strptime(eval(date1), _date_format)
    _d2 = datetime.strptime(eval(date2), _date_format)
    
    if _d2 > _d1:
        return int(1)
    else:
        return int(0)

@problog_export('+str', '+str', '-float')
def diff_dates(date1, date2):
    _date_format = "%Y-%m-%d %H:%M:%S.%f"

    _d1 = datetime.strptime(eval(date1), _date_format)
    _d2 = datetime.strptime(eval(date2), _date_format)

    return float((_d2 - _d1).total_seconds())

@problog_export_nondet('+str', '+str')
def compare_dates_bool(date1, date2):
    _date_format = "%Y-%m-%d %H:%M:%S.%f"

    _d1 = datetime.strptime(eval(date1), _date_format)
    _d2 = datetime.strptime(eval(date2), _date_format)
    
    if _d1 > _d2:
        return [()]
    else:
        return []
