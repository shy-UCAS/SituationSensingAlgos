% [RULES] juding enemy's distances to our important facilities
approach_facility(EUav, Facility) :-
    dist_to_facility(EUav, Facility, shrink),
    angle_to_facility(EUav, Facility, narrow).

dist_to_facility(EUav, Facility, shrink) :-
    dist(EUav, Facility, D1, T1), dist(EUav, Facility, D2, T2),
    D2 < D1 * (1 - 0.1), T1 < T2.

dist_to_facility(EUav, Facility, grow) :-
    dist(EUav, Facility, D1, T1), dist(EUav, Facility, D2, T2),
    D2 > D1 * (1 + 0.1), T1 < T2.

dist_to_facility(EUav, Facility, stable) :-
    dist(EUav, Facility, D1, T1), dist(EUav, Facility, D2, T2),
    D2 =< D1 * (1 + 0.1), D2 >= D1 * (1 - 0.1), T1 < T2.

% [RULES] juding enemy's angles to our important facilities
angle_to_facility(EUav, Facility, narrow) :-
    move_inter_angle(EUav, Facility, A),
    A =< 20.

angle_to_facility(EUav, Facility, wide) :-
    move_inter_angle(EUav, Facility, A),
    A > 20.

% [RULES] juding enemy's targeting facility
targeting_facility(EUav, Facility) :-
    angle_to_facility(EUav, Facility, narrow),
    approach_facility(EUav, Facility).

0.9::move_inter_angle(euav1, hq1, 15).

query(targeting_facility(_, _)).
query(angle_to_facility(_, _, _)).