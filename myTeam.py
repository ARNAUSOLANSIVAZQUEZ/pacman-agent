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
                first='ToxicReflexAgent', second='ProtectionReflexAgent', num_training=0):
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


class ToxicReflexAgent(ReflexCaptureAgent):
    # Defining adaptive agent that attacks keeping a victory margin to stay ahead. If we are close to lose
    # or we already surpass the margin it falls back to defense. Mirrors defensive behaviour of Protect agent
    # except it doesn't care about the power pill.
    def get_features(self, game_state, action):
        # Initializing features counter
        features = util.Counter()

        # Retrieving successor state
        successor = self.get_successor(game_state, action)

        # Retrieving own/enemy food list for successor state
        enemy_food = self.get_food(successor).as_list()
        my_food = self.get_food_you_are_defending(successor).as_list()

        # Retrieving own state and position for successor state
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Retrieving enemy states for successor state
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Retrieving score for current game state
        score = self.get_score(game_state)

        # If we are not winning by a margin of 5 toggle offensive behaviour
        if score <= 5:
            # Defining winning margin based on next food lists
            win_margin = len(my_food) - len(enemy_food)
            # If we are not 5 foods ahead toggle offensive behaviour
            # and If we are not close to losing (only 2 or less food remaining on our side)
            if win_margin < 5 and len(my_food) > 2:
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
                    food_distances = [self.get_maze_distance(my_pos, food) for food in enemy_food]
                    min_food_distance = min(food_distances)
                    # Define objective distance as distance from closer enemy food
                    features['objective_distance'] = min_food_distance
                    # If agent is on enemy field make it fearful of enemies
                    if my_state.is_pacman:
                        # Compute visible non-scared enemy ghosts distances
                        enemy_positions = []
                        for a in enemies:
                            if not(a.is_pacman) and a.scared_timer == 0:
                                position = a.get_position()
                                if position is not None:
                                    enemy_positions.append(position)
                        # If any enemy is visible compute minimum distance between agent and enemy
                        if len(enemy_positions) > 0:
                            enemy_distances = [self.get_maze_distance(my_pos, enemy_pos) for enemy_pos in enemy_positions]
                            min_enemy_distance = min(enemy_distances)
                            # Define fear proportionally inverse to closer ghost
                            features['fear'] = 1/min_enemy_distance
            # Else toggle defensive behaviour
            else:
                # Toggle defense
                features['on_defense'] = 1
                features['num_invaders'] = len(invaders)
                # Nullify offensive feature
                features['fear'] = 0
                features['successor_score'] = 0
                # If agent is on enemy field make it fearful of enemies (returning)
                if my_state.is_pacman:
                    # Compute visible non-scared enemy ghosts distances
                    enemy_positions = []
                    for a in enemies:
                        if not (a.is_pacman) and a.scared_timer == 0:
                            position = a.get_position()
                            if position is not None:
                                enemy_positions.append(position)
                    # If any enemy is visible compute minimum distance between agent and enemy
                    if len(enemy_positions) > 0:
                        enemy_distances = [self.get_maze_distance(my_pos, enemy_pos) for enemy_pos in enemy_positions]
                        min_enemy_distance = min(enemy_distances)
                        # Define fear proportionally inverse to closer ghost
                        features['fear'] = 1 / min_enemy_distance
                # If there are invaders compute minimum distance to invader
                if len(invaders) > 0:
                    invader_distances = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                    min_invader_distance = min(invader_distances)
                    # Define objective distance as minimum distance from agent to invader
                    features['objective_distance'] = min_invader_distance
                # Else compute distance to own food
                else:
                    my_food_distances = [self.get_maze_distance(my_pos, food) for food in my_food]
                    min_my_food_distance = min(my_food_distances)
                    # Define objective distance as minimum distance from agent to own food
                    features['objective_distance'] = min_my_food_distance
        # If we are winning with a margin of 5 toggle defense
        else:
            features['on_defense'] = 1
            features['successor_score'] = 0
            features['fear'] = 0
            features['num_invaders'] = len(invaders)
            # If there are invaders compute minimum distance to invader
            if len(invaders) > 0:
                invader_distances = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                min_invader_distance = min(invader_distances)
                # Define objective distance as minimum distance from agent to invader
                features['objective_distance'] = min_invader_distance
            # Else compute distance to own food
            else:
                my_food_distances = [self.get_maze_distance(my_pos, food) for food in my_food]
                min_my_food_distance = min(my_food_distances)
                # Define objective distance as minimum distance from agent to own food
                features['objective_distance'] = min_my_food_distance
        # Controlling passive behaviour based on if the agent is on the offense or defense
        # Defensive behaviour allows agent to stop on top of ally food
        if features['on_defense'] == 1:
            if action == Directions.STOP:
                features['stop'] = 0
        # Aggressive behaviour forces agent to keep moving towards current objective
        elif features['on_defense'] == 0:
            if action == Directions.STOP:
                features['stop'] = 1
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': -100, 'on_defense': 100,
                'objective_distance': -1, 'num_invaders': -1000,
                'stop': -100, 'fear': -10}


class ProtectionReflexAgent(ReflexCaptureAgent):
    # Defining fully defensive agent to protect our side with priority:
    # Invader > Power Pill > Own Food
    def get_features(self, game_state, action):
        # Initializing features counter
        features = util.Counter()

        # Retrieving successor state
        successor = self.get_successor(game_state, action)

        # Retrieving own food list for successor state
        my_food = self.get_food_you_are_defending(successor).as_list()
        # Retrieving own power pill for successor state
        my_power = self.get_capsules_you_are_defending(game_state)

        # Retrieving own state and position for successor state
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Retrieving enemy states for successor state
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] == 0
        features['num_invaders'] = len(invaders)
        if action == Directions.STOP:
            features['stop'] = 0
        # If there are invaders compute minimum distance to invader
        if len(invaders) > 0:
            invader_distances = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            min_invader_distance = min(invader_distances)
            # Define objective distance as minimum distance from agent to invader
            features['objective_distance'] = min_invader_distance
        # Else if there are own power pills compute minimum distance to power pill
        elif len(my_power) > 0:
            my_power_distances = [self.get_maze_distance(my_pos, power) for power in my_power]
            min_my_power_distance = min(my_power_distances)
            # Define objective distance as minimum distance from agent to power pill
            features['objective_distance'] = min_my_power_distance
        # Else compute distances to own food
        else:
            my_food_distances = [self.get_maze_distance(my_pos, food) for food in my_food]
            min_my_food_distance = min(my_food_distances)
            # Define objective distance as minimum distance from agent to own food
            features['objective_distance'] = min_my_food_distance
        return features

    def get_weights(self, game_state, action):
        return {'on_defense': 100, 'objective_distance': -1,
                'num_invaders': -1000, 'stop': -100}
