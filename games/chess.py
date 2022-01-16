from collections import deque
import datetime
import os

import numpy as np
import torch
import chess

from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (19, 8, 8)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(len(POLICY_UCI)))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 8  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 10  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 5  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 20  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 32  # Number of channels in reward head
        self.reduced_channels_value = 32  # Number of channels in value head
        self.reduced_channels_policy = 32  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1e5  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.2  # Initial learning rate
        self.lr_decay_rate = .1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Chess()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
    
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                action = UCI_TO_POLICY[str(
                    input("Enter your move in uci format")
                )]
                
                if action in self.legal_actions():
                    break
            except:
                pass
            print("Not a legal move, try again")
        return action

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, uci_move):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play {uci_move}"

class Chess:
    def __init__(self, history_size: int = 8):
        self.board = chess.Board()
        self.player = 1
    
    def to_play(self):
        return int(not self.player)

    def reset(self):
        self.board = chess.Board()
        self.player = 1
        return self.get_observation()

    def step(self, action: int):

        self.board.push_uci(POLICY_UCI[int(action)])

        done = self.board.is_game_over()

        reward = self.get_reward()

        self.player = 1 - self.player
        self.board.apply_mirror()

        return self.get_observation(), reward, done

    def get_reward(self):
        outcome = self.board.outcome()
        if outcome:
            result = outcome.result()

            if result == '1-0':
                return 1
            if result == '0-1':
                return -1
            else:
                return 0

        else:
            return 0

    def get_observation(self):
        return board_to_planes(self.board, self.player)

    def legal_actions(self):
        return [UCI_TO_POLICY[move.uci()] for move in self.board.legal_moves]

    def expert_action(self):
        return UCI_TO_POLICY[np.random.choice(list(self.board.legal_moves)).uci()]

    def have_outcome(self):
        return True if self.board.is_game_over() else False

    def render(self):
        print(self.board)

def board_to_planes(board, to_play):
    board_fen, _, castling_rights, _, half_moves, full_moves = board.fen().split(" ")
    piece_planes = get_piece_planes(board_fen)
    special_planes = get_special_planes(castling_rights, to_play, half_moves, full_moves)
        
    return np.concatenate([piece_planes, special_planes])

def get_piece_planes(board_fen):
    pieces = "KQBNRPkqbnrp"
    fen = replace_empty_squares(board_fen)
    
    fen = fen.replace("/", "")
    fen = np.array(list(fen)).reshape((8,8))
    planes = np.zeros((12,8,8))
    for i, piece in enumerate(pieces):
        planes[i][np.where(fen == piece)] = 1
        
    return planes

def get_special_planes(castling_rights, to_play, half_moves, full_moves):
    planes = []
    
    # castling rights
    for right in "KQkq":
        planes.append(np.full((8,8), int(right in castling_rights)))
        
    # half moves
    planes.append(np.full((8,8), int(half_moves)))
    
    # full moves
    planes.append(np.full((8,8), int(full_moves)))
    
    # player color
    planes.append(np.full((8,8), to_play))
    
    planes = np.array(planes)
    
    return planes

def replace_empty_squares(board_fen: str):
    for i in range(1,9):
        board_fen = board_fen.replace(str(i), i*"0")
        
    return board_fen

def empty_planes(planes: np.ndarray):
    return np.zeros(planes.shape)

def get_move_to_index():
    codes, i = {}, 0
    directions = np.array([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]])

    for direction in directions:
        for squares in range(1,8):
            codes[tuple(list(squares * direction) + [None])] = i
            i += 1

    for two in [2, -2]:
        for one in [1, -1]:
            codes[(two, one, None)], i = i, i+1

    for two in [2, -2]:
        for one in [1,-1]:
            codes[(one, two, None)], i = i, i+1

    for move in [[0,1],[1,1],[-1,1]]:
        for promote_to in ["q", "r", "n", "b"]:
            codes[tuple(move + [promote_to])] , i = i , i + 1
            
    return codes

def get_policy_uci():
    index_to_file = dict(enumerate("abcdefgh"))
    file_to_index = {v:k for k,v in index_to_file.items()}
    assert len(index_to_file) == len(file_to_index)

    index_to_rank = dict(enumerate(range(1,9)))
    rank_to_index = {v:k for k,v in index_to_rank.items()}
    assert len(index_to_rank) == len(rank_to_index)

    move_to_index = get_move_to_index()
    index_to_move = {v:k for k,v in move_to_index.items()}
    assert len(move_to_index) == len(index_to_move)

    policy_uci = np.zeros((8,8,len(move_to_index)), object)

    for file in range(8):
        for rank in range(8):
            for move_index, move in index_to_move.items():            
                to_file = file + move[0]
                to_rank = rank + move[1]
                
                promotion = move[2] if move[2] else ''
                
                uci_move = \
                    index_to_file[file] + str(index_to_rank[rank]) + \
                    index_to_file.get(to_file, '_') + str(index_to_rank.get(to_rank, '_')) + \
                    promotion
                
                policy_uci[file][rank][move_index] = uci_move

    policy_uci = policy_uci.flatten()

    return policy_uci

POLICY_UCI = get_policy_uci()
UCI_TO_POLICY = {v:k for k,v in enumerate(POLICY_UCI.flatten()) if '_' not in v} 