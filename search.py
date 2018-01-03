# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, PriorityQueue

from multiprocessing import Queue, Process
from collections import deque
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]


def tree_search(problem,fringe):
    fringe.push((problem.getStartState(),"",0))
    from game import Directions
    while fringe:
        node = fringe.pop()

        state = node[0]
        direction = node[1]
        print("Directions."+direction)
        if problem.isGoalState(state):
            return direction
        children = problem.getSuccessors(state)
        for child in children:
            fringe.push(child)

    return None


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:


    """
    #use list like a queue
    frontier = []
    explored = set()

    frontier.append(Node(problem.getStartState()))
    while frontier:
        node = frontier.pop()

        if problem.isGoalState(node.state):
            return node.path()
        explored.add(node.state)

        children = problem.getSuccessors(node.state)
        for child in children:
            #rebuild successor attributes into a node
            c = Node(child[0],child[1],child[2],node)
            if child[0] not in explored and child not in frontier:
                frontier.append(c)
    return None





    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    root = Node(problem.getStartState())
    if problem.isGoalState(root.state):
        return root


    frontier = deque([root])
   # frontier.append(node)
    explored = set()

    while frontier:

        node = frontier.popleft()
        if problem.isGoalState(node.state):
            return node.path()
        explored.add(node.state)
        children = problem.getSuccessors(node.state)
       # if problem.isGoalState(node.state):
         #   return node.path()
        #do this sooner

        trail = []
       # print node.state
        for link in frontier:
            trail.append(link.state)
        for child in children:
            c = Node(child[0], child[1], child[2], node)
            if child[0] not in explored and child[0] not in trail:
                frontier.append(c)

    return None
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    #I am now fond of using lists for data structures...
    frontier = []
    root = Node(problem.getStartState())
    frontier.append((root.cost,root))
    explored = set()
    while frontier:
        frontier.sort(reverse=True)
        x, y = frontier.pop()
        node = y
        if problem.isGoalState(node.state):
            return node.path()
        explored.add(y.state)
        children = problem.getSuccessors(node.state)
        for child in children:
            print child[1], child[2] + node.cost
           # parent = Node(node.state, node.actionFromParent, node.cost, node.parent)
            c = Node(child[0], child[1], child[2]+node.cost, node)
            trail = []
            for tuple in frontier:
                x, y = tuple
                trail.append(y.state)

            if child[0] not in explored and child[0] not in trail:
                frontier.append((c.cost,c))
            elif child[0] in trail:
                for tuple in frontier:
                    x,y = tuple
                    if child[0]==y.state and c.cost<y.cost:
                        #since we dont know the index, iterate through frontier until the node is found
                        #in hindsight, this implementation might contribute to slowdown...
                        del tuple
                        frontier.append((c.cost,c))
                        #stop when this is done
                        break




    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    node = Node(problem.getStartState())
    frontier = PriorityQueue()
    explored = set()
    frontier.push(node,node.cost+heuristic(node.state,problem))
    temp = []
    temp.append(node)
    while not frontier.isEmpty():

        helper = PriorityQueue()
        #since we cannot iterate through PriorityQueue() the way we did in UCS...
        while not frontier.isEmpty():
            x = frontier.pop()
            h = heuristic(x.state,problem)
            temp.append(x.state)
            helper.push(x,x.cost+h)
        #memoization wannabe
        while not helper.isEmpty():
            y = helper.pop()
            h = heuristic(y.state,problem)
            frontier.push(y,y.cost+h)

        node = frontier.pop()
        explored.add(node.state)
        if problem.isGoalState(node.state):
            return node.path()

        children = problem.getSuccessors(node.state)
        for (child,action,cost) in children:
            #algorithm is identical to UCS, save for heuristic affecting priority
            nc = Node(child,action,node.cost+cost,node)
            h = heuristic(child,problem)
            if child not in explored and child not in temp:
                frontier.push(nc,nc.cost+h)

            else :
                while not frontier.isEmpty():
                    x = frontier.pop()
                    e = heuristic(x.state,problem)
                    if x.state==child and nc.cost<x.cost:
                        helper.push(nc,nc.cost+h)
                    else:
                        helper.push(x,x.cost+e)
            while not helper.isEmpty():
                y = helper.pop()
                h = heuristic(y.state,problem)
                frontier.push(y,y.cost+h)






    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
#_________________
#node class as discussed in slides
class Node:
    def __init__(self, state, action=None, cost=0.0, parent=None):
        self.actions = []  # e.g., to store full path from root
        self.actionFromParent = action
        self.cost = cost
        self.state = state
        self.parent = parent
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
        # more code here if you want to maintain actions. need special case for root

#constructs the path that lead to goalState
    def path(self):
        node = self
        path = []
        while node:
            path.append(node.actionFromParent)
            node = node.parent
        path.pop()
        return list(reversed(path))

