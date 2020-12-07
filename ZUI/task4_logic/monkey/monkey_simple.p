%#######################
%DESCRIPTION
% this is the basic solution for the monkey-banana problem.
% In this case, only monkey, banana and box exist.
% This solution uses world as the only description of the situation at one time
% world(__,__,__,__,__,__) -> world(__,__,__,__,__,__).
% syntax world(box_position, banana_position, monkey_position, monkey_vertical_position, have_banana, state).

fof(init_state, axiom,
  (world(roh, stred, stred, dole, ne, s0))
).

fof(diff_roh_stred, axiom,
  (roh != stred)
).

fof(diff_ano_ne, axiom,
  (ano != ne)
).

fof(diff_nahore_dole, axiom,
  (nahore != dole)
).
%#######################
% ACTIONS

%push (we can push anywhere we want, the other possible
% solution that would be more efficient (in this simple case)
% would be move the box only under the banana).
% Monkey can push only when is down (dole).
fof(push_action, axiom,
  (! [X1, X2, Y, W, P] : (
    (world(X1,Y,X1,dole,P,W) & ( X1 != X2 ) )
    => world(X2,Y,X2,dole,P,vysledek(posun(X1,X2),W))
  ) )
).

%move
fof(move_action, axiom,
  (! [X, Y, Z1, Z2, W, P] : (
    ( world(X,Y,Z1,dole,P,W) & ( Z1 != Z2 ) )
      => world(X,Y,Z2,dole,P,vysledek(pohyb(Z1,Z2),W))
  ) )
).

%climb-up
fof(climb_up_action, axiom,
  (! [X, Y, Z, W, P] : (
    world(X,Y,X,dole,P,W)
    => world(X,Y,X,nahore,P,vysledek(vylezt(dole,nahore),W))
  ) )
).

% tear off the banana (the monkey has to be on the box under the banana)
fof(tore_banana_action, axiom,
  (! [W] : (
    world(stred,stred,stred,nahore,ne,W)
    => world(stred,stred,stred,nahore,ano,vysledek(tore(banana),W))
  ) )
).

% the goal to be proven
fof(problem_goal,conjecture,
  (? [W, X, Y, Z, P]: (
    world(X,Y,Z,P,ano,W)
  ) )
).
