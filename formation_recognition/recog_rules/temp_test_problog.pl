:- use_module('census_infer.py').

date('2024-12-09 13:45:32.312').
date('2024-12-09 13:45:33.791').

date_later(Date1, Date2) :-
    date(Date1), date(Date2), Date1 \= Date2, compare_dates_int(Date2, Date1, LaterBool), LaterBool = 1.

query(date_later(_, _)).
query(diff_dates('2024-12-09 13:45:32.312', '2024-12-09 13:45:33.791', DiffSecs)).
