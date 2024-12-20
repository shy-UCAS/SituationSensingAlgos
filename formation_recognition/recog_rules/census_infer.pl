% ProbB::fact_b.
% ProbC::fact_c.
% ProbD::fact_d.

% 0.5::fact_d.

fact_a :- fact_b.
fact_a :- fact_c.
fact_a :- fact_d.

% fact_a :- fact_b; fact_c; fact_d.

% query(fact_a).

% [RULES] juding enemy's distances to our important facilities
date_later(Date1, Date2) :- compare_dates(Date1, Date2).

approach_facility(EUav, Facility) :-
    dist_to_facility(EUav, Facility, shrink),
    angle_to_facility(EUav, Facility, narrow).

dist_to_facility(EUav, Facility, shrink) :-
    avg_dist(EUav, Facility, D1, T1), avg_dist(EUav, Facility, D2, T2),
    D2 < D1 * (1 - VarDistChangeEps), date_later(T2, T1).

dist_to_facility(EUav, Facility, grow) :-
    avg_dist(EUav, Facility, D1, T1), avg_dist(EUav, Facility, D2, T2),
    D2 > D1 * (1 + VarDistChangeEps), date_later(T2, T1).

dist_to_facility(EUav, Facility, stable) :-
    avg_dist(EUav, Facility, D1, T1), avg_dist(EUav, Facility, D2, T2),
    D2 =< D1 * (1 + VarDistChangeEps), D2 >= D1 * (1 - VarDistChangeEps), T1 < T2.

% [RULES] juding enemy's angles to our important facilities
targeting_facility(EUav, Facility) :-
    angle_to_facility(EUav, Facility, narrow).

angle_to_facility(EUav, Facility, narrow) :-
    move_inter_angle(EUav, Facility, A),
    A =< VarInterAngleEps.

angle_to_facility(EUav, Facility, wide) :-
    move_inter_angle(EUav, Facility, A),
    A > VarInterAngleEps.

% [RULES] enemy directing angle fluctuations
angle_to_facility_fluctlevel(EUav, Facility, High) :-
    angle_to_facility_fluctstd(EUav, Facility, FluctStd),
    FluctStd > VarAngleFluctStdEps.

angle_to_facility_fluctlevel(EUav, Facility, Low) :-
    angle_to_facility_fluctstd(EUav, Facility, FluctStd),
    FluctStd =< VarAngleFluctStdEps.
