%--------------------------------------------------------------------------
% Generalized monkey-banana problem: 
% A monkey, dog and cat are in a room. A banana hangs from the ceiling, it is beyond the animals' reach. 
% The monkey is able to walk, move and climb objects. The cat cannot move objects, the dog cannot climb objects. 
% The objects in the room can be climbable and movable, in particular, a box is both climbable and movable, a picture is movable and a toilet is climbable.
% The room is just the right height so that an animal who moves an object under the banana and climbs it can grasp the banana. Of course, the animal has to have the abilities, the object has to have the needed properties.
% The goal is to generate a sucessful plan (i.e., a sequence of simple actions) automatically.
% The monkey can get to the banana on its own, the dog and cat need to cooperate.

% Runtime:
% The full task (3 agents, 3 objects): plan_any conjecture proven in 50 seconds, the monkey uses the box
% In the cooperative design (the plan reached by dog and cat), simplify the task formalization to get a solution below 100s.
%  - remove redundant agents objects (plan_cat satisfiable without monkey, toilet, picture)
%  - minimize the number of variables in the conjecture
%--------------------------------------------------------------------------

% Simple animal/object properties
% Inequalities to substitute for missing UNA

fof(diff_door_corner, axiom,
    (door != corner)
). 

fof(diff_door_below_banana, axiom,
    (door != below_banana)
). 

fof(diff_corner_below_banana, axiom,
    (corner != below_banana)
). 

fof(prop_box, axiom,
    (climbable(box))
). 

fof(prop_toilet, axiom,
    (climbable(toilet))
). 

fof(prop_painting, axiom,
    (~climbable(painting))
). 

fof(prop_box, axiom,
    (pushable(box))
). 

fof(prop_toilet, axiom,
    (~pushable(toilet))
). 

fof(prop_painting, axiom,
    (pushable(painting))
). 

fof(diff_box_toilet, axiom,
    (box != toilet)
). 

fof(diff_box_painting, axiom,
    (box != painting)
). 

fof(diff_toilet_painting, axiom,
    (toilet != painting)
). 

fof(prop_monkey, axiom,
    (climbs(monkey))
). 

fof(prop_cat, axiom,
    (climbs(cat))
). 

fof(prop_dog, axiom,
    (~climbs(dog))
). 

fof(prop_monkey, axiom,
    (pushes(monkey))
). 

fof(prop_cat, axiom,
    (~pushes(cat))
). 

fof(prop_dog, axiom,
    (pushes(dog))
). 

fof(diff_ground_air, axiom,
    (ground != air)
). 

fof(diff_with_without_banana, axiom,
    (with_banana != without_banana)
). 

fof(diff_monkey_nobody, axiom,
    (monkey != nobody)
). 

fof(diff_monkey_dog, axiom,
    (monkey != dog)
). 

fof(diff_monkey_cat, axiom,
    (monkey != cat)
). 

fof(diff_dog_nobody, axiom,
    (dog != nobody)
). 

fof(diff_dog_cat, axiom,
    (dog != cat)
). 

fof(diff_cat_nobody, axiom,
    (cat != nobody)
). 

% Actions	

% go	
fof(go_agent_effect, axiom,
    ![A,P1,P2,B,S]: (
	(P1 != P2 &
	 agent(A, P1, B, ground, S)
	)
	=> agent(A, P2, B, ground, result(go(A, P2), S))
    )
).

fof(go_agent_frame, axiom,
    ![A1,A2,P1,P2,P3,B1,B2,X,S]: (
	(P1 != P2 &
	 agent(A1, P1, B1, ground, S) &
	 agent(A2, P3, B2, X, S) &
	 A1 != A2
	)
	=> agent(A2, P3, B2, X, result(go(A1, P2), S))
    )
).

fof(go_object_frame, axiom,
    ![A1,A2,P1,P2,P3,B,I,S]: (
	(P1 != P2 &
	 agent(A1, P1, B, ground, S) &
	 item(I, P3, A2, S)
	)
	=> item(I, P3, A2, result(go(A1, P2), S))
    )
).

% push
fof(push_agent_item_effect, axiom,
    ![P1,P2,A,S,I]: (
	(P1 != P2 &
	 pushes(A) & 
	 agent(A, P1, without_banana, ground, S) & 
	 pushable(I) & 
	 item(I, P1, nobody, S)
	) 
	=> (agent(A, P2, without_banana, ground, result(push(A, I, P1, P2), S)) &
		item(I, P2, nobody, result(push(A, I, P1, P2), S)))
    )
).

fof(push_agent_frame, axiom,
    ![P1,P2,P3,A1,A2,S,I,B,X]: (
	(P1 != P2 &
	 pushes(A1) & 
	 agent(A1, P1, without_banana, ground, S) & 
	 pushable(I) & 
	 item(I, P1, nobody, S) &
	 agent(A2, P3, B, X, S) &
	 A1 != A2
	) 
	=> agent(A2, P3, B, X, result(push(A1, I, P1, P2), S))
    )
).

fof(push_item_frame, axiom,
    ![P1,P2,P3,A1,A2,S,I1,I2]: (
	(P1 != P2 &
	 pushes(A1) & 
	 agent(A1, P1, without_banana, ground, S) & 
	 pushable(I1) & 
 	 item(I1, P1, nobody, S) &
	 item(I2, P3, A2, S) &
	 I1 != I2
	) 
	=> item(I2, P3, A2, result(push(A1, I1, P1, P2), S))
    )
).

fof(climb_agent_item_effect, axiom,
    ![A,P,B,S,I]: (
	(climbs(A) & 
	 agent(A, P, B, ground, S) &
	 climbable(I) &
	 item(I, P, nobody, S)
	) 
	=> (
	agent(A, P, B, air, result(climb_up(A, I), S)) &
	item(I, P, A, result(climb_up(A, I), S))
	)
    )
).

fof(climb_agent_frame, axiom,
    ![A1,A2,P1,P2,B1,B2,S,I,X]: (
	(climbs(A1) & 
	 agent(A1, P1, B1, ground, S) &
	 climbable(I) &
	 item(I, P1, nobody, S) &
	 agent(A2, P2, B2, X, S) &
    	 A1 != A2	
	) 
	=> agent(A2, P2, B2, X, result(climb_up(A1, I), S))
    )
).

fof(climb_item_frame, axiom,
    ![A1,A2,P1,P2,B,S,I1,I2]: (
	(climbs(A1) & 
	 agent(A1, P1, B, ground, S) &
	 climbable(I1) &
	 item(I1, P1, nobody, S) &
	 item(I2, P2, A2, S) &
	 I1 != I2
	) 
	=> item(I2, P2, A2, result(climb_up(A1, I1), S))
    )
).

fof(grasp_agent_effect, axiom,
    ![A,S,I]: (    
	(agent(A, below_banana, without_banana, air, S) & item(I, below_banana, A, S)) 
	=> 
	 agent(A, below_banana, with_banana, air, result(grasp(A), S))
    )
).

fof(grasp_agent_frame, axiom,
    ![A1,A2,S,I,P,B,X]: (
	(agent(A1, below_banana, without_banana, air, S) & 
	 item(I, below_banana, A1, S) & 
	 agent(A2, P, B, X, S) &
	 A1 != A2
	)
	=> 
	 agent(A2, P, B, X, result(grasp(A1), S))
    )
).

fof(grasp_item_frame, axiom,
    ![A1,A2,S,I,I1,P]: (
	(agent(A1, below_banana, without_banana, air, S) & 
	 item(I, below_banana, A1, S) & 
	 item(I1, P, A2, S)
	)
	=> 
	 item(I1, P, A2, result(grasp(A1), S))
    )
).
	
% Initial world setting

fof(init_box, axiom,
    item(box, corner, nobody, s0)
). 

fof(init_toilet, axiom,
    item(toilet, corner, nobody, s0)
). 

fof(init_painting, axiom,
    item(painting, corner, nobody, s0)
). 

fof(init_monkey, axiom,
    agent(monkey, door, without_banana, ground, s0)
). 

fof(init_cat, axiom,
    agent(cat, door, without_banana, ground, s0)
). 

fof(init_dog, axiom,
    agent(dog, door, without_banana, ground, s0)
). 

% Conjectures to be proven

fof(plan_monkey, conjecture,
    ?[S]: (agent(monkey, below_banana, with_banana, air, S))
).

%fof(plan_any, conjecture,
%    ?[A,S]: (agent(A, below_banana, with_banana, air, S))
%).
	
%fof(plan_cat, conjecture,
%    ?[X,Y,S]: (agent(cat, X, with_banana, Y, S))
%).
