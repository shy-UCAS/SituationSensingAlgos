0.948::flying_speed(euav0, medium).
0.581::slow_down(euav0).
0.843::change_direction(euav0, large).
0.8::direct_fluctuate(euav0, low).

0.9::targeting_facility(euav0, none, 19.000).
0.8::distance_to_facility(euav0, radar_1, closing).
0.8::distance_to_facility(euav0, hq_1, closing).
0.8::distance_to_facility(euav0, hq_2, closing).
0.8::distance_to_facility(euav0, ua_1, closing).
0.8::distance_to_facility(euav0, ua_2, closing).
0.793::flying_speed(euav1, fast).
0.700::steady_speed(euav1).
0.737::change_direction(euav1, large).
0.8::direct_fluctuate(euav1, low).
0.9::probed_facility(euav1, hq_2, 8.000).

0.8::distance_to_facility(euav1, radar_1, closing).
0.8::distance_to_facility(euav1, hq_1, closing).
0.8::distance_to_facility(euav1, hq_2, closing).
0.8::distance_to_facility(euav1, ua_1, closing).
0.8::distance_to_facility(euav1, ua_2, closing).
0.904::flying_speed(euav2, medium).
0.700::steady_speed(euav2).
0.8::change_direction(euav2, small).
0.8::direct_fluctuate(euav2, low).
0.9::probed_facility(euav2, ua_2, 13.000).
0.9::targeting_facility(euav2, none, 19.000).
0.8::distance_to_facility(euav2, radar_1, closing).
0.8::distance_to_facility(euav2, hq_1, closing).
0.8::distance_to_facility(euav2, hq_2, closing).
0.8::distance_to_facility(euav2, ua_1, closing).
0.8::distance_to_facility(euav2, ua_2, closing).
0.864::flying_speed(euav3, fast).
0.845::speed_up(euav3). 
0.8::change_direction(euav3, small).
0.8::direct_fluctuate(euav3, low).


0.8::distance_to_facility(euav3, radar_1, closing).
0.8::distance_to_facility(euav3, hq_1, closing).
0.8::distance_to_facility(euav3, hq_2, closing).
0.8::distance_to_facility(euav3, ua_1, closing).
0.8::distance_to_facility(euav3, ua_2, closing).
0.843::flying_speed(euav4, fast).
0.700::steady_speed(euav4).
0.8::change_direction(euav4, small).
0.8::direct_fluctuate(euav4, low).

0.8::distance_to_facility(euav4, radar_1, closing).
0.8::distance_to_facility(euav4, hq_1, closing).
0.8::distance_to_facility(euav4, hq_2, closing).
0.8::distance_to_facility(euav4, ua_1, closing).
0.8::distance_to_facility(euav4, ua_2, closing).
1.0::defence_facility(ua_1). 1.0::defence_facility(ua_2). 1.0::defence_facility(ua_3). 
1.0::defence_facility(radar_1). 1.0::defence_facility(radar_2). 1.0::defence_facility(radar_3).

1.0::important_facility(ua_1). 1.0::important_facility(ua_2). 1.0::important_facility(ua_3). 
1.0::important_facility(hq_1). 1.0::important_facility(hq_2). 1.0::important_facility(hq_3).
1.0::important_facility(radar_1). 1.0::important_facility(radar_2). 1.0::important_facility(radar_3).

1.0:: non_defensive_facility(hq_1). 1.0:: non_defensive_facility(hqhq_22). 1.0:: non_defensive_facility(hq_3).

% [Rules] recognizing penetration intention
single_penetration(EUav) :-
    targeting_facility(EUav, I_Facility, _), important_facility(I_Facility),
    distance_to_facility(EUav, D_Facility, closing), defence_facility(D_Facility),
    flying_speed(EUav, high),
    tight_fleet(EUav).

% [Rules] recognizing reconnaisance intention
single_reconnaisance(EUav) :-
    direct_fluctuate(EUav, high), 
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

sequential_attack(EUav) :-
    attack_same_facility(EUavs, Times), is_member_of(EUav, EUavs),
    change_direction(Euav, small),
    sequential_time(Times).

% [Rules] recognizing salvo attacks
salvo_attack(EUav) :-
    attack_same_facility(EUavs, Times), is_member_of(EUav, EUavs),
    change_direction(EUav, small),
    salvo_time(Times).

% querying conclusions
query(single_penetration(EUav)).
query(single_reconnaisance(EUav)).
query(single_detouring(EUav)).
query(single_fast_passing(EUav)).