
% Enter your KB below this line:

suggest(rest_0) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(japanese), alcohol(serves), take_away(not_offers), coffee(not_serves), pescaterian(friendly), open(early), price_range(w7500_15000).
suggest(rest_1) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(korean), alcohol(not_serves), take_away(offers), coffee(not_serves), pescaterian(friendly), open(early), price_range(w5000_7500).
suggest(rest_2) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(taiwanese), alcohol(serves), take_away(not_offers), coffee(not_serves), pescaterian(friendly), open(early), price_range(w15000_25000).
suggest(rest_3) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(korean), alcohol(not_serves), take_away(offers), coffee(not_serves), pescaterian(not_friendly), open(late), price_range(w5000_7500).
suggest(rest_4) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(korean), alcohol(serves), take_away(not_offers), coffee(not_serves), pescaterian(not_friendly), open(late), price_range(w7500_15000).
suggest(rest_5) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(italian), alcohol(serves), take_away(not_offers), coffee(serves), pescaterian(friendly), open(late), price_range(w15000_25000).
suggest(rest_6) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(korean), alcohol(not_serves), take_away(not_offers), coffee(serves), pescaterian(not_friendly), open(early), price_range(w7500_15000).
suggest(rest_7) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(breakfast), alcohol(serves), take_away(not_offers), coffee(serves), pescaterian(friendly), open(late), price_range(w7500_15000).
suggest(rest_8) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(vietnamese), alcohol(serves), take_away(not_offers), coffee(not_serves), pescaterian(not_friendly), open(w24/7), price_range(w7500_15000).
suggest(rest_9) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(vietnamese), alcohol(serves), take_away(not_offers), coffee(not_serves), pescaterian(not_friendly), open(late), price_range(w7500_15000).
suggest(rest_10) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(korean), alcohol(not_serves), take_away(not_offers), coffee(not_serves), pescaterian(not_friendly), open(early), price_range(w7500_15000).
suggest(rest_11) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), alcohol(not_serves), take_away(offers), coffee(not_serves), pescaterian(not_friendly), open(w24/7), price_range(w5000_7500).
suggest(rest_12) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(vietnamese), alcohol(not_serves), take_away(offers), coffee(not_serves), pescaterian(not_friendly), open(early), price_range(w5000_7500).
suggest(rest_13) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), alcohol(serves), take_away(not_offers), coffee(not_serves), pescaterian(not_friendly), open(late), price_range(w25000+).
suggest(rest_14) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(american), alcohol(serves), take_away(offers), coffee(serves), pescaterian(not_friendly), open(early), price_range(w7500_15000).
suggest(rest_15) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), alcohol(serves), take_away(offers), coffee(not_serves), pescaterian(friendly), open(early), price_range(w7500_15000).
suggest(rest_16) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), alcohol(not_serves), take_away(offers), coffee(not_serves), pescaterian(friendly), open(early), price_range(w7500_15000).
suggest(rest_17) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(american), alcohol(not_serves), take_away(offers), coffee(serves), pescaterian(friendly), open(w24/7), price_range(w7500_15000).
suggest(rest_18) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(american), alcohol(not_serves), take_away(offers), coffee(serves), pescaterian(friendly), open(w24/7), price_range(w7500_15000).
suggest(rest_19) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(korean), alcohol(serves), take_away(offers), coffee(not_serves), pescaterian(not_friendly), open(early), price_range(w7500_15000).
suggest(rest_20) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(chinese), alcohol(serves), take_away(offers), coffee(not_serves), pescaterian(friendly), open(early), price_range(w7500_15000).
suggest(rest_21) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(korean), alcohol(serves), take_away(offers), coffee(not_serves), pescaterian(friendly), open(late), price_range(w7500_15000).
suggest(rest_22) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(korean), alcohol(serves), take_away(not_offers), coffee(not_serves), pescaterian(not_friendly), open(w24/7), price_range(w7500_15000).
suggest(rest_23) :- neighborhood(itaewon), vegeterian(not_friendly), cuisine_type(turkish), alcohol(not_serves), take_away(offers), coffee(not_serves), pescaterian(not_friendly), open(early), price_range(w5000_7500).
suggest(rest_24) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(american), alcohol(not_serves), take_away(offers), coffee(serves), pescaterian(friendly), open(late), price_range(w15000_25000).
suggest(rest_25) :- neighborhood(gangnam), vegeterian(not_friendly), cuisine_type(korean), alcohol(serves), take_away(not_offers), coffee(not_serves), pescaterian(not_friendly), open(early), price_range(w7500_15000).
suggest(rest_26) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(korean), alcohol(serves), take_away(offers), coffee(not_serves), pescaterian(friendly), open(w24/7), price_range(w5000_7500).
suggest(rest_27) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), alcohol(not_serves), take_away(offers), coffee(not_serves), pescaterian(friendly), open(early), price_range(w7500_15000).

neighborhood(X) :- check(neighborhood, X).
vegeterian(X) :- check(vegeterian, X).
cuisine_type(X) :- check(cuisine_type, X).
alcohol(X) :- check(alcohol, X).
take_away(X) :- check(take_away, X).
coffee(X) :- check(coffee, X).
pescaterian(X) :- check(pescaterian, X).
open(X) :- check(open, X).
price_range(X) :- check(price_range, X).

multivalued(neighborhood).
multivalued(cuisine_type).
multivalued(price_range).
multivalued(open).
neighborhood(X) :- check(neighborhood, X).
vegeterian(X) :- check(vegeterian, X).
cuisine_type(X) :- check(cuisine_type, X).
alcohol(X) :- check(alcohol, X).
take_away(X) :- check(take_away, X).
coffee(X) :- check(coffee, X).
pescaterian(X) :- check(pescaterian, X).
open(X) :- check(open, X).
price_range(X) :- check(price_range, X).

multivalued(neighborhood).
multivalued(cuisine_type).
multivalued(price_range).
multivalued(open).

% Care-checking implemented to allow the user to ignore attributes they don't care about:

check(A, X) :- not(check_if_care(A, X)).
check_if_care(A, X) :- care(A), not(ask(A, X)).
care(A) :- atom_concat(care_check_, A, X), ask(X, '').

% Asking clauses

ask(A, V):-
known(y, A, V), % succeed if true
!. % stop looking

ask(A, V):-
known(_, A, V), % fail if false
!, fail.

ask(A, V):-
not(multivalued(A)),
% write_py(A:not_multivalued),
known(y, A, V2),
V \== V2,
!, fail.

ask(A, V):-
read_py(A,V,Y), % get the answer
asserta(known(Y, A, V)), % remember it
Y == y. % succeed or fail
