0.840::flying_speed(euav0, fast).
0.700::steady_speed(euav0).
0.8::change_direction(euav0, small).
0.8::direct_fluctuate(euav0, low).
0.9::probed_facility(euav0, hq_2, 22.000).
0.9::probed_facility(euav0, ua_2, 24.000).

0.8::distance_to_facility(euav0, radar_1, closing).
0.8::distance_to_facility(euav0, hq_1, staying).
0.8::distance_to_facility(euav0, hq_2, closing).
0.8::distance_to_facility(euav0, ua_1, closing).
0.8::distance_to_facility(euav0, ua_2, closing).
0.829::flying_speed(euav1, fast).
0.700::steady_speed(euav1).
0.912::change_direction(euav1, large).
0.8::direct_fluctuate(euav1, low).

0.9::targeting_facility(euav1, none, 29.000).
0.8::distance_to_facility(euav1, radar_1, closing).
0.8::distance_to_facility(euav1, hq_1, staying).
0.8::distance_to_facility(euav1, hq_2, closing).
0.8::distance_to_facility(euav1, ua_1, staying).
0.8::distance_to_facility(euav1, ua_2, closing).
0.826::flying_speed(euav2, fast).
0.700::steady_speed(euav2).
0.8::change_direction(euav2, small).
0.8::direct_fluctuate(euav2, low).
0.9::probed_facility(euav2, hq_2, 19.000).

0.8::distance_to_facility(euav2, radar_1, staying).
0.8::distance_to_facility(euav2, hq_1, staying).
0.8::distance_to_facility(euav2, hq_2, staying).
0.8::distance_to_facility(euav2, ua_1, staying).
0.8::distance_to_facility(euav2, ua_2, staying).
0.886::flying_speed(euav3, fast).
0.864::speed_up(euav3).
0.893::change_direction(euav3, large).
0.8::direct_fluctuate(euav3, low).
0.9::probed_facility(euav3, hq_1, 15.000).
0.9::probed_facility(euav3, ua_1, 14.000).

0.8::distance_to_facility(euav3, radar_1, closing).
0.8::distance_to_facility(euav3, hq_1, staying).
0.8::distance_to_facility(euav3, hq_2, staying).
0.8::distance_to_facility(euav3, ua_1, staying).
0.8::distance_to_facility(euav3, ua_2, staying).
0.848::flying_speed(euav4, fast).
0.700::steady_speed(euav4).
0.8::change_direction(euav4, small).
0.8::direct_fluctuate(euav4, low).
0.9::probed_facility(euav4, radar_1, 26.000).
0.534::targeting_facility(euav4, hq_1, 12.000).
0.701::targeting_facility(euav4, hq_2, 12.000).
0.412::targeting_facility(euav4, radar_1, 13.000).
0.8::distance_to_facility(euav4, radar_1, closing).
0.8::distance_to_facility(euav4, hq_1, closing).
0.8::distance_to_facility(euav4, hq_2, closing).
0.8::distance_to_facility(euav4, ua_1, closing).
0.8::distance_to_facility(euav4, ua_2, closing).
0.9::attack_same_facility([], [], none, none).

1.0::defence_facility(ua_1). 1.0::defence_facility(ua_2). 1.0::defence_facility(ua_3).
1.0::defence_facility(radar_1). 1.0::defence_facility(radar_2). 1.0::defence_facility(radar_3).

1.0::important_facility(ua_1). 1.0::important_facility(ua_2). 1.0::important_facility(ua_3).
1.0::important_facility(hq_1). 1.0::important_facility(hq_2). 1.0::important_facility(hq_3).
1.0::important_facility(radar_1). 1.0::important_facility(radar_2). 1.0::important_facility(radar_3).

1.0::non_defensive_facility(hq_1). 1.0::non_defensive_facility(hqhq_22). 1.0::non_defensive_facility(hq_3).

% [Rules] recognizing penetration intention
single_penetration(EUav) :-
    targeting_facility(EUav, Facility, _),
    distance_to_facility(EUav, D_Facility, closing), defence_facility(D_Facility),
    flying_speed(EUav, high).

% [Rules] recognizing reconnaisance intention
single_reconnaisance(EUav) :-
    (direct_fluctuate(EUav, high); change_direction(EUav, large)),
    (distance_to_facility(EUav, Facility, staying); distance_to_facility(EUav, Facility, away_from)),
    important_facility(Facility).

% [Rules] recognizing detouring intention
single_detouring(EUav) :-
    (distance_to_facility(EUav, Facility, staying); distance_to_facility(EUav, Facility, away_from)),
    defence_facility(Facility),
    (change_direction(EUav, large); change_direction(EUav, medium)).

% [Rules] recognizing fast passing intention
single_fast_passing(EUav) :-
    change_direction(EUav, small), flying_speed(EUav, high),
    (targeting_facility(EUav, none, _); targeting_facility(EUav, Facility, _), non_defensive_facility(Facility)).

% [Rules] recognizing search & strike
search_in_searchstrike(EUav) :-
    single_reconnaisance(EUav),
    probed_facility(EUav, Facility, Tsearch), targeting_facility(EUavB, Facility, Ttarget),
    Tsearch < Ttarget.

strike_in_searchstrike(EUav) :-
    targeting_facility(EUav, Facility, Ttarget),
    single_reconnaisance(EUavA), probed_facility(EUavA, Facility, Tsearch),
    Tsearch < Ttarget.

% [Rules] recognizing sequential / salvo attacks
1.0::is_member_of(X, [X|_]).
is_member_of(X, [_|L]) :- is_member_of(X, L).

% find_all_times(EUav, [Tl|T]) :-

sequential_attack(EUav, Facility) :-
    attack_same_facility(EUavs, Times, Facility, sequential), is_member_of(EUav, EUavs),
    change_direction(Euav, small).

% [Rules] recognizing salvo attacks
salvo_attack(EUav, Facility) :-
    attack_same_facility(EUavs, Times, Facility, same_time), is_member_of(EUav, EUavs),
    change_direction(EUav, small).

% % test: hypothesized knowledges
% 0.8::targeting_facility(euav1, hq_1).
% 0.9::distance_to_facility(euav1, radar_1, closing).
% 0.7::distance_to_facility(euav1, hq_1, away_from).
% 0.3::direct_fluctuate(euav1).
% 0.9::change_direction(euav1, small).
% 0.6::flying_speed(euav1, high).
% 0.7::tight_fleet(euav1).

% 0.7::targeting_facility(euav2, ua_1).
% 0.8::distance_to_facility(euav2, ua_1, closing).
% 0.6::distance_to_facility(euav2, hq_1, away_from).
% 0.9::direct_fluctuate(euav2).
% 0.95::change_direction(euav2, large).
% 0.3::flying_speed(euav2, high).
% 0.1::tight_fleet(euav2).

% querying conclusions
query(single_penetration(EUav)).
query(single_reconnaisance(EUav)).
query(single_detouring(EUav)).
query(single_fast_passing(EUav)).
query(sequential_attack(EUav, Fac)).
query(salvo_attack(EUav,Fac)).