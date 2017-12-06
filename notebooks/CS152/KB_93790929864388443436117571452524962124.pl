
run_command(TOKENS, COMMAND) :-
   command(CLIST, TOKENS, []),
   COMMAND =.. CLIST,
   call(COMMAND).

% Define the exit predicate to do nothing.
exit.

%%% DCGs

command([OP|ARGS]) --> operation(OP), arguments(ARGS).

arguments([ARG|ARGS]) --> argument(ARG), arguments(ARGS).
arguments([]) --> [].

operation(report) --> [list].
operation(book) --> [book].
operation(cancel) --> [cancel].
operation(exit) --> ([exit]; [quit]; [bye]).

argument(passengers) --> [passengers].
argument(flights) --> [flights].

argument(FLIGHT) --> [FLIGHT], {flight(FLIGHT)}.
argument(PASSENGER) --> [PASSENGER].

% Flights

flight(aa101).
flight(sq238).
flight(mi436).
flight(oz521).

% Command predicates

report(flights) :-
   flight(F),
   write_py(F),
   fail.
report(_).

report(passengers, FLIGHT) :-
   booked(PASSENGER, FLIGHT),
   write_py(PASSENGER),
   fail.
report(_, _).

book(PASSENGER, FLIGHT) :-
   assertz(booked(PASSENGER, FLIGHT)).

cancel(PASSENGER, FLIGHT) :-
   retract(booked(PASSENGER, FLIGHT)).

%%%%% End Prolog File
