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
    attack_same_facility(EUavs, Times, sequential), is_member_of(EUav, EUavs),
    change_direction(Euav, small).

% [Rules] recognizing salvo attacks
salvo_attack(EUav) :-
    attack_same_facility(EUavs, Times, same_time), is_member_of(EUav, EUavs),
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
query(sequential_attack(EUav)).
query(salvo_attack(EUav)).