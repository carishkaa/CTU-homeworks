%---------------------------------------------------------------
% Hanoi towers
%---------------------------------------------------------------
% task description: https://en.wikipedia.org/wiki/Tower_of_Hanoi
% three pegs (a,b,c), two discs (sizes 1,2), move discs from a to b, keep constraints
% predicate: peg(T,U,W,S) three lists maintain the current state of the world,
% action: move(X,Y1,Y2) move disc whose size is X from the peg Y1 to the peg Y2.
% in this solution, sitcalc not necessary, it only records the actions, computationally more efficient than the parallel solution
%---------------------------------------------------------------
% Theory:

% size constraints, which disk can be placed where

fof(d1_ineq,hypothesis,
  d1!=void
).

fof(d2_ineq,hypothesis,
  d2!=void
).

fof(d3_ineq,hypothesis,
  d3!=void
).

fof(d1_on_ground,hypothesis,
  can_place(d1,void)
).

fof(d2_on_ground,hypothesis,
  can_place(d2,void)
).

fof(d3_on_ground,hypothesis,
  can_place(d3,void)
).

fof(d1_on_d2,hypothesis,
  can_place(d1,d2)
).

fof(d1_on_d3,hypothesis,
  can_place(d1,d3)
).

fof(d2_on_d3,hypothesis,
  can_place(d2,d3)
).
  
% initial world
fof(initial,hypothesis,
  peg([d1,d2,d3],[void,void,void],[void,void,void],s0)
).
  
% move disc from first to second
fof(move_disc_from_first_to_second,axiom,
  ! [X1,X2,X3,Y1,Y2,Y3,Z,S]: (
    (peg([X1,X2,X3],[Y1,Y2,Y3],Z,S) & can_place(X1,Y1) & X1!=void) => peg([X2,X3,void],[X1,Y1,Y2],Z,result(move(X1,a,b),S))
  )
).

% move disc from first to third
fof(move_disc_from_first_to_third,axiom,
  ! [X1,X2,X3,Y,Z1,Z2,Z3,S]: (
    (peg([X1,X2,X3],Y,[Z1,Z2,Z3],S) & can_place(X1,Z1) & X1!=void) => peg([X2,X3,void],Y,[X1,Z1,Z2],result(move(X1,a,c),S))
  )
).

% move disc from second to third
fof(move_disc_from_second_to_third,axiom,
  ! [X,Y1,Y2,Y3,Z1,Z2,Z3,S]: (
    (peg(X,[Y1,Y2,Y3],[Z1,Z2,Z3],S) & can_place(Y1,Z1) & Y1!=void) => peg(X,[Y2,Y3,void],[Y1,Z1,Z2],result(move(Y1,b,c),S))
  )
).

% move disc from second to first
fof(move_disc_from_second_to_first,axiom,
  ! [X1,X2,X3,Y1,Y2,Y3,Z,S]: (
    (peg([X1,X2,X3],[Y1,Y2,Y3],Z,S) & can_place(Y1,X1) & Y1!=void) => peg([Y1,X1,X2],[Y2,Y3,void],Z,result(move(Y1,b,a),S))
  )
).

% move disc from third to first
fof(move_disc_from_third_to_first,axiom,
  ! [X1,X2,X3,Y,Z1,Z2,Z3,S]: (
    (peg([X1,X2,X3],Y,[Z1,Z2,Z3],S) & can_place(Z1,X1) & Z1!=void) => peg([Z1,X1,X2],Y,[Z2,Z3,void],result(move(Z1,c,a),S))
  )
).

% move disc from third to second
fof(move_disc_from_third_to_second,axiom,
  ! [X,Y1,Y2,Y3,Z1,Z2,Z3,S]: (
    (peg(X,[Y1,Y2,Y3],[Z1,Z2,Z3],S) & can_place(Z1,Y1) & Z1!=void) => peg(X,[Z1,Y1,Y2],[Z2,Z3,void],result(move(Z1,c,b),S))
  )
).

%---------------------------------------------------------------
% Conjectures/tests (use only one of them):
   
% one step proof
fof(one_step_proof,conjecture,
  ? [S]: peg([d2,d3,void],[void,void,void],[d1,void,void],S)
).  

% discs cannot double, cannot be proven
fof(sanity_proof,conjecture,
  ? [S]: peg([d2,d3,void],[d1,void,void],[d1,void,void],S)
).  

% order must be kept, cannot be proven
fof(sanity_proof,conjecture,
  ? [X,Y,Z,S]: (peg([void,X,void],Y,Z,S) & X!=void)
).  

% discs cannot swap, cannot be proven
fof(sanity_proof,conjecture,
  ? [X,Y,Z,S]: peg(X,[d2,d1,void],Z,S)
).  

% final proof
fof(final,conjecture,
  ? [S]: peg([void,void,void],[d1,d2,d3],[void,void,void],S)
).
