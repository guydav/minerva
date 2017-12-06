
% Enter your KB below this line:

:- dynamic(known/3). 

suggest(rest_0) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(japanese), alcohol(served), not(take_away(offered)), not(coffee(served)), pescaterian(friendly), open(early), price_range(w7500_15000).
suggest(rest_1) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(korean), not(alcohol(served)), take_away(offered), not(coffee(served)), pescaterian(friendly), open(early), price_range(w5000_7500).
suggest(rest_2) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(taiwanese), alcohol(served), not(take_away(offered)), not(coffee(served)), pescaterian(friendly), open(early), price_range(w15000_25000).
suggest(rest_3) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(korean), not(alcohol(served)), take_away(offered), not(coffee(served)), not(pescaterian(friendly)), open(late), price_range(w5000_7500).
suggest(rest_4) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(korean), alcohol(served), not(take_away(offered)), not(coffee(served)), not(pescaterian(friendly)), open(late), price_range(w7500_15000).
suggest(rest_5) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(italian), alcohol(served), not(take_away(offered)), coffee(served), pescaterian(friendly), open(late), price_range(w15000_25000).
suggest(rest_6) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(korean), not(alcohol(served)), not(take_away(offered)), coffee(served), not(pescaterian(friendly)), open(early), price_range(w7500_15000).
suggest(rest_7) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(breakfast), alcohol(served), not(take_away(offered)), coffee(served), pescaterian(friendly), open(late), price_range(w7500_15000).
suggest(rest_8) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(vietnamese), alcohol(served), not(take_away(offered)), not(coffee(served)), not(pescaterian(friendly)), open(w24/7), price_range(w7500_15000).
suggest(rest_9) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(vietnamese), alcohol(served), not(take_away(offered)), not(coffee(served)), not(pescaterian(friendly)), open(late), price_range(w7500_15000).
suggest(rest_10) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(korean), not(alcohol(served)), not(take_away(offered)), not(coffee(served)), not(pescaterian(friendly)), open(early), price_range(w7500_15000).
suggest(rest_11) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), not(alcohol(served)), take_away(offered), not(coffee(served)), not(pescaterian(friendly)), open(w24/7), price_range(w5000_7500).
suggest(rest_12) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(vietnamese), not(alcohol(served)), take_away(offered), not(coffee(served)), not(pescaterian(friendly)), open(early), price_range(w5000_7500).
suggest(rest_13) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), alcohol(served), not(take_away(offered)), not(coffee(served)), not(pescaterian(friendly)), open(late), price_range(w25000+).
suggest(rest_14) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(american), alcohol(served), take_away(offered), coffee(served), not(pescaterian(friendly)), open(early), price_range(w7500_15000).
suggest(rest_15) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), alcohol(served), take_away(offered), not(coffee(served)), pescaterian(friendly), open(early), price_range(w7500_15000).
suggest(rest_16) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), not(alcohol(served)), take_away(offered), not(coffee(served)), pescaterian(friendly), open(early), price_range(w7500_15000).
suggest(rest_17) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(american), not(alcohol(served)), take_away(offered), coffee(served), pescaterian(friendly), open(w24/7), price_range(w7500_15000).
suggest(rest_18) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(american), not(alcohol(served)), take_away(offered), coffee(served), pescaterian(friendly), open(w24/7), price_range(w7500_15000).
suggest(rest_19) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(korean), alcohol(served), take_away(offered), not(coffee(served)), not(pescaterian(friendly)), open(early), price_range(w7500_15000).
suggest(rest_20) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(chinese), alcohol(served), take_away(offered), not(coffee(served)), pescaterian(friendly), open(early), price_range(w7500_15000).
suggest(rest_21) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(korean), alcohol(served), take_away(offered), not(coffee(served)), pescaterian(friendly), open(late), price_range(w7500_15000).
suggest(rest_22) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(korean), alcohol(served), not(take_away(offered)), not(coffee(served)), not(pescaterian(friendly)), open(w24/7), price_range(w7500_15000).
suggest(rest_23) :- neighborhood(itaewon), not(vegeterian(friendly)), cuisine_type(turkish), not(alcohol(served)), take_away(offered), not(coffee(served)), not(pescaterian(friendly)), open(early), price_range(w5000_7500).
suggest(rest_24) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(american), not(alcohol(served)), take_away(offered), coffee(served), pescaterian(friendly), open(late), price_range(w15000_25000).
suggest(rest_25) :- neighborhood(gangnam), not(vegeterian(friendly)), cuisine_type(korean), alcohol(served), not(take_away(offered)), not(coffee(served)), not(pescaterian(friendly)), open(early), price_range(w7500_15000).
suggest(rest_26) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(korean), alcohol(served), take_away(offered), not(coffee(served)), pescaterian(friendly), open(w24/7), price_range(w5000_7500).
suggest(rest_27) :- neighborhood(gangnam), vegeterian(friendly), cuisine_type(japanese), not(alcohol(served)), take_away(offered), not(coffee(served)), pescaterian(friendly), open(early), price_range(w7500_15000).

neighborhood(X) :- ask(neighborhood, X).
vegeterian(X) :- ask(vegeterian, X).
cuisine_type(X) :- ask(cuisine_type, X).
alcohol(X) :- ask(alcohol, X).
take_away(X) :- ask(take_away, X).
coffee(X) :- ask(coffee, X).
pescaterian(X) :- ask(pescaterian, X).
open(X) :- ask(open, X).
price_range(X) :- ask(price_range, X).

multivalued(neighborhood).
multivalued(cuisine_type).
multivalued(price_range).
multivalued(open).

% The code below implements the prompting to ask the user:


% Asking clauses

multivalued(none).

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
