:- use_module('census_infer.py').

% [RULES] checking enemy's speed changes
speed_up(EUav, Timestamp2) :- 
    avg_speed(EUav, FlySpeed1, TimeStamp1), avg_speed(EUav, FlySpeed2, Timestamp2), 
    FlySpeed2 > FlySpeed1 * (1 + VarSpeedChangeEps), 
    date_later_than(Timestamp2, TimeStamp1).

slow_down(EUav, Timestamp2) :-
    avg_speed(EUav, FlySpeed1, TimeStamp1), avg_speed(EUav, FlySpeed2, Timestamp2), 
    FlySpeed2 < FlySpeed1 * (1 - VarSpeedChangeEps), 
    date_later_than(Timestamp2, TimeStamp1).

% [RULES] moving direct angle fluctuation analyze
orient_change(EUav, DiffAngle, Timestamp2) :-
    avg_orient(EUav, Orient1, TimeStamp1), avg_orient(EUav, Orient2, Timestamp2),
    date_later_than(Timestamp2, TimeStamp1),
    infer_orient_diff(Orient1, Orient2, DiffAngle),
    DiffAngle > VarInterAngleEps.

orient_nochange(EUav, DiffAngle, Timestamp2) :-
    avg_orient(EUav, Orient1, TimeStamp1), avg_orient(EUav, Orient2, Timestamp2),
    date_later_than(Timestamp2, TimeStamp1),
    infer_orient_diff(Orient1, Orient2, DiffAngle),
    DiffAngle =< VarInterAngleEps.

infer_orient_diff(Orient1, Orient2, DiffAngle) :-
    OrigDiff is mod(abs(Orient2 - Orient1), 360),
    adjust_diff_angle(OrigDiff, DiffAngle).

adjust_diff_angle(Angle, Result) :- Angle > 180, Result is 360 - Angle.
adjust_diff_angle(Angle, Result) :- Angle =< 180, Result is Angle.

% [RULES] juding enemy's distances to our important facilities
date_later_than(Date1, Date2) :- Date1 \= Date2, compare_dates_bool(Date1, Date2).

approach_facility(EUav, Facility) :-
    dist_to_facility(EUav, Facility, shrink),
    angle_to_facility(EUav, Facility, narrow).

dist_to_facility(EUav, Facility, shrink) :-
    avg_dist(EUav, Facility, D1, T1), avg_dist(EUav, Facility, D2, T2),
    D2 < D1 * (1 - VarDistChangeEps), date_later_than(T2, T1).

dist_to_facility(EUav, Facility, grow) :-
    avg_dist(EUav, Facility, D1, T1), avg_dist(EUav, Facility, D2, T2),
    D2 > D1 * (1 + VarDistChangeEps), date_later_than(T2, T1).

dist_to_facility(EUav, Facility, stable) :-
    avg_dist(EUav, Facility, D1, T1), avg_dist(EUav, Facility, D2, T2),
    D2 =< D1 * (1 + VarDistChangeEps), D2 >= D1 * (1 - VarDistChangeEps), T1 < T2.

% [RULES] juding enemy's angles to our important facilities
targeting_facility(EUav, Facility) :-
    angle_to_facility(EUav, Facility, narrow).

angle_to_facility(EUav, Facility, narrow) :-
    move_to_facility_angle(EUav, Facility, A),
    A =< VarInterAngleEps.

angle_to_facility(EUav, Facility, wide) :-
    move_to_facility_angle(EUav, Facility, A),
    A > VarInterAngleEps.

% [RULES] enemy directing angle fluctuations
angle_to_facility_fluctlevel(EUav, Facility, High) :-
    angle_to_facility_fluctstd(EUav, Facility, FluctStd),
    FluctStd > VarAngleFluctStdEps.

angle_to_facility_fluctlevel(EUav, Facility, Low) :-
    angle_to_facility_fluctstd(EUav, Facility, FluctStd),
    FluctStd =< VarAngleFluctStdEps.
