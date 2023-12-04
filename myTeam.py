# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='StallReflexAgent', second='StallReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class StallReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        # Initializing features counter
        features = util.Counter()

        # Retrieving successor state
        successor = self.get_successor(game_state, action)
        # Retrieving own/enemy food list for successor state
        enemy_food = self.get_food(successor).as_list()
        my_food = self.get_food_you_are_defending(successor).as_list()

        # Retrieving own/enemy food list for current state
        cur_enemy_food = self.get_food(game_state).as_list()
        cur_my_food = self.get_food(game_state).as_list()

        # Retrieving own state and position for successor state
        my_state = successor.get_agent_state(self.index)
        my_pos = successor.get_agent_state(self.index).get_position()
        # Retrieving enemy states for successor state
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Retrieving score for current game state
        score = self.get_score(game_state)

        # If we are not winning by a margin of 5 toggle offensive behaviour
        if score <= 5:
            # Defining winning margin based on current food
            win_margin = len(my_food) - len(enemy_food)
            print(win_margin)
            # If we are not 5 foods ahead toggle offensive behaviour // If we are not going to lose
            if win_margin < 5 and len(my_food) != 2:
                print('OFFENSE')
                # Toggle offense
                features['on_defense'] = 0
                features['successor_score'] = len(enemy_food)
                # Agent is not fearful until it sees enemy ghosts
                features['fear'] = 0
                # Nullify defensive feature
                features['num_invaders'] = 0
                # Check if there is remaining enemy food
                if len(enemy_food) > 0:
                    # Retrieving minimum distance to enemy food
                    min_distance = min([self.get_maze_distance(my_pos, food) for food in enemy_food])
                    # Compute visible enemy ghosts distances
                    enemy_positions = []
                    for a in enemies:
                        if not a.is_pacman:
                            position = a.get_position()
                            if position is not None:
                                enemy_positions.append(position)
                    # If agent is on enemy field make it fearful of enemies
                    if my_state.is_pacman:
                        # If any enemy is visible compute minimum distance between agent and enemy
                        if len(enemy_positions) > 0:
                            min_enemy_distance = min([self.get_maze_distance(my_pos, enemy_pos) for enemy_pos in enemy_positions])
                            features['fear'] = 1/min_enemy_distance
                    # Define objective distance as a ratio between food distance and enemy distance
                    features['objective_distance'] = min_distance
            # Else toggle defensive behaviour
            else:
                print('DEFENSE')
                # Toggle defense
                features['on_defense'] = 1
                features['num_invaders'] = len(invaders)
                # Nullify offensive feature
                features['fear'] = 0
                features['successor_score'] = 0
                # Compute visible enemy ghosts distances
                enemy_positions = []
                for a in enemies:
                    if not a.is_pacman:
                        position = a.get_position()
                        if position is not None:
                            enemy_positions.append(position)
                # If agent is on enemy field make it fearful of enemies (returning)
                if my_state.is_pacman:
                    # If any enemy is visible compute minimum distance between agent and enemy
                    if len(enemy_positions) > 0:
                        min_enemy_distance = min([self.get_maze_distance(my_pos, enemy_pos) for enemy_pos in enemy_positions])
                        features['fear'] = 1 / min_enemy_distance
                    # If there are invaders compute minimum distance to invader
                    if len(invaders) > 0:
                        min_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders])
                        # Define objective distance as minimum distance from agent to invader
                        features['objective_distance'] = min_distance
                    # Else compute distance to own food
                    else:
                        min_distance = min([self.get_maze_distance(my_pos, food) for food in my_food])
                        # Define objective distance as minimum distance from agent to own food
                        features['objective_distance'] = min_distance
        # If we are winning with a margin of 5 toggle defense
        else:
            print('WINNING')
            features['on_defense'] = 1
            features['successor_score'] = 0
            features['fear'] = 0
            features['num_invaders'] = len(invaders)
            # If there are invaders compute minimum distance to invader
            if len(invaders) > 0:
                    min_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders])
                    # Define objective distance as minimum distance from agent to invader
                    features['objective_distance'] = min_distance
            # Else compute distance to own food
            else:
                    min_distance = min([self.get_maze_distance(my_pos, food) for food in my_food])
                    # Define objective distance as minimum distance from agent to own food
                    features['objective_distance'] = min_distance
        # Controlling passive behaviour based on if the agent is on the offense or defense
        if features['on_defense'] == 1:
            if action == Directions.STOP:
                features['stop'] = 0
        elif features['on_defense'] == 0:
            if action == Directions.STOP:
                features['stop'] = 1
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': -100, 'on_defense': 100,
                'objective_distance': -1, 'num_invaders': -1000,
                'stop': -100, 'fear': -10}

class DefenseReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        # Initializing features counter
        features = util.Counter()
        # Retrieving successor state
        successor = self.get_successor(game_state, action)
        my_food = self.get_food_you_are_defending(successor).as_list()
        # Retrieving own state and position for successor state
        my_state = successor.get_agent_state(self.index)
        my_pos = successor.get_agent_state(self.index).get_position()
        # Retrieving enemy states for successor state
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        # Initializing features for number of invaders and score of successor state
        features['num_invaders'] = len(invaders)
        # Initializing current behaviour (offense/defense)
        features['on_defense'] = 1
        # If there are invaders compute minimum distance to invader
        if len(invaders) > 0:
            min_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders])
            # Define objective distance as minimum distance from agent to invader
            features['objective_distance'] = min_distance
        # Else compute distance to own food
        else:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in my_food])
            # Define objective distance as minimum distance from agent to own food
            features['objective_distance'] = min_distance
        # Controlling passive behaviour based on if the agent is on the offense or defense
        if features['on_defense'] == 1:
            if action == Directions.STOP:
                features['stop'] = 0
        elif features['on_defense'] == 0:
            if action == Directions.STOP:
                features['stop'] = 1
        return features

    def get_weights(self, game_state, action):
        return {'on_defense': 100, 'objective_distance': -1,
                'num_invaders': -1000, 'stop': -100}