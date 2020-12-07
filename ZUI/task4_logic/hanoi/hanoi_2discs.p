%---------------------------------------------------------------
% Hanoi towers
%---------------------------------------------------------------
% task description: https://en.wikipedia.org/wiki/Tower_of_Hanoi
% a simplified version, two discs only to proof in the order of seconds, can easily be extended  
% three pegs (a,b,c), two discs (sizes 1,2), move discs from a to b, keep constraints
% axiom: loc(X,Y,P,S) the disc whose size is X is on the peg Y, there is P other discs above it, the axiom refers to the world S
% action: move(X,Y1,Y2) move disc whose size is X from the peg Y1 to the peg Y2.
%---------------------------------------------------------------
% Theory:

% inequalities, constants refer to different objects
fof(a_b_different,hypothesis,
  a!=b
).

fof(a_c_different,hypothesis,
  a!=c
).

fof(b_c_different,hypothesis,
  b!=c
).

fof(d1_infty_different,hypothesis,
  d1!=infty
).

fof(d2_infty_different,hypothesis,
  d2!=infty
).

fof(d1_d2_different,hypothesis,
  d1!=d2
).

% size constraints, cannot place larger disc on a smaller one  
fof(d1_infty_can_place,hypothesis,
  can_place(d1,infty)
).

fof(d2_infty_can_place,hypothesis,
  can_place(d2,infty)
).

fof(d1_d2_can_place,hypothesis,
  can_place(d1,d2)
).

% initial world
fof(init_ground_peg_a,hypothesis,
  loc(infty,a,s(s(none)),s0)
).

fof(init_ground_peg_b,hypothesis,
  loc(infty,b,none,s0)
).

fof(init_ground_peg_c,hypothesis,
  loc(infty,c,none,s0)
).

fof(init_d1_peg_a,hypothesis,
  loc(d1,a,none,s0)
).

fof(init_d2_peg_a,hypothesis,
  loc(d2,a,s(none),s0)
).

% effect axioms
fof(move_a_disc,axiom,
  ! [D1,D2,K1,K2,S]: (
    (loc(D1,K1,none,S) & loc(D2,K2,none,S) & can_place(D1,D2)) 
       => loc(D1,K2,none,result(move(D1,K1,K2),S)))
).

% frame axiom, the current move does not effect the given disc
fof(only_copy_disc,axiom,
  ! [D1,D2,K1,K2,K3,P,S]:(
    (loc(D1,K1,P,S) & D1!=D2 & K1!=K2 & K1!=K3)
       => loc(D1,K1,P,result(move(D2,K2,K3),S)))
).

fof(remove_disc_above,axiom,
  ! [D1,D2,K1,K2,P,S]:(
    (loc(D1,K1,s(P),S) & can_place(D2,D1) & K1!=K2)
       => loc(D1,K1,P,result(move(D2,K1,K2),S)))
).

% a slower option for the previous, final proof in about 90s
%fof(remove_disc_above,axiom,
%  ! [D1,D2,K1,K2,P,S]:(
%    (loc(D1,K1,s(P),S) & loc(D2,K2,none,result(move(D2,K1,K2),S)) & D1!=D2)
%       => loc(D1,K1,P,result(move(D2,K1,K2),S)))
%).

fof(add_disc_above,axiom,
  ! [D1,D2,K1,K2,P,S]:(
    (loc(D1,K2,P,S) & can_place(D2,D1) & K1!=K2)
       => loc(D1,K2,s(P),result(move(D2,K1,K2),S)))
).

% a slower option for the previous, final proof in about 90s
%fof(add_disc_above,axiom,
%  ! [D1,D2,K1,K2,P,S]:(
%    (loc(D1,K2,P,S) & loc(D2,K2,none,result(move(D2,K1,K2),S)) & D1!=D2)
%       => loc(D1,K2,s(P),result(move(D2,K1,K2),S)))
%).

%---------------------------------------------------------------
% Conjectures/tests (use only one of them):
% test1 and test2 altogether make the complete solution
% test1: only change the conjecture, keep the init unchanged
% test1 reached in about 0.02 seconds
fof(test1_positive,conjecture,
  ?[S]: (loc(d2,a,none,S) & loc(d1,c,none,S))
).

% test2: the conjecture the same as in the final test, but init changed as follows (remove the original init state definition)
% test2 reached in about 0.03 seconds
fof(init_ground_peg_a,hypothesis,
  loc(infty,a,s(none),s0)
).

fof(init_ground_peg_b,hypothesis,
  loc(infty,b,none,s0)
).

fof(init_ground_peg_c,hypothesis,
  loc(infty,c,s(none),s0)
).

fof(init_d1_peg_a,hypothesis,
  loc(d1,c,none,s0)
).

fof(init_d2_peg_a,hypothesis,
  loc(d2,a,none,s0)
).

% the complete solution, about 4s to be reached
fof(testf_positive,conjecture,
  ?[S]: (loc(d2,b,s(none),S))
).

% a couple of negative proofs, they cannot be proven
fof(test3_negative,conjecture,
  ?[P,S]: (loc(infty,b,none,S) & loc(d2,b,P,S))
).

fof(test4_negative,conjecture,
  ?[P,S]: (loc(infty,a,none,S) & loc(infty,b,none,S) & loc(infty,c,none,S))
).
