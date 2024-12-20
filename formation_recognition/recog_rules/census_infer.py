from datetime import datetime
from problog.extern import problog_export

@problog_export('+str', '+str', '-bool')
def compare_dates(date1, date2):
    _date_format = "%Y-%m-%d %H:%M:%S.%f"
    _d1 = datetime.strptime(date1, _date_format)
    _d2 = datetime.strptime(date2, _date_format)
    return _d1 > _d2