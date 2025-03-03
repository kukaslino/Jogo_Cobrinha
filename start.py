from agent import Agent;
from game import SnakeGameAI;
from graph import performance_graph;

def train():
    graph_scores = []
    graph_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            graph_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            graph_mean_scores.append(mean_score)
            performance_graph(graph_scores, graph_mean_scores)

if __name__ == '__main__':
    train()