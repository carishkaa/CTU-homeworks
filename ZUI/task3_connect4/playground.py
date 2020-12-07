from agent import DeterministicAgent
from testing_games import TreeGame, create_tree_game_score
from utils import extend_game_with_counter

def run_single_tree_estimate():
    '''
    This is simple function demonstrating the correctness of your algorithm. You can use it to debug your code.
    Ultimatelly, we would recommend to write you to write your own pytests.
    '''
    depth = 3
    branching_factor = 3

    # Creates a game with depth 4 and unspecified terminal values - it will 
    # throw an error if we try to get the terminal value
    # This is ok, since we will evaluate our heuristic function at depth 3 
    # and should never go deeper than that
    game = TreeGame(depth + 1, branching_factor, None)

    # Creates the heuristic function associated with this game
    # We will use a tree with depth 3 and branching factor 3, so it can be debugged easily
    values = [-11,-10, 17, 12,-8,14,16,-2,-2,-1,14,-17,0,-5,0,8,3,10,-9,8,10,-5,-13,-15,12,-11,-11]
    heuristic_fn = create_tree_game_score(depth, values)

    # Here we will extend the game with a counter to measure the performance of our algorithm
    # Feel free to comment these lines if you dont want to measure the performance of your algorithm
    # and want your debugging to be easier
    TreeGameMonitored, heuristic_fn = extend_game_with_counter(TreeGame, heuristic_fn)
    game = TreeGameMonitored(depth + 1, branching_factor, None)

    # Construct the instance of your agent
    agent = DeterministicAgent(heuristic_fn, game.to_play, depth)

    # Compute the game value recursively
    value = agent.compute_estimate(game.clone())

    # Please ensure yourself this is the correct value of the game
    print(f"The game value is: {value}")
    assert value == 14, "game value should be 14"

    if hasattr(game, 'estimated_nodes_count'):
        print(f"Expanded nodes: {game.expanded_nodes_count}, heuristic fn calls: {game.estimated_nodes_count}")

if __name__ == "__main__":
    run_single_tree_estimate()
