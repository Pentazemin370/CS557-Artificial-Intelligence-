# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor)
        foodCoordinates = newFood.asList()
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #for ghostState in newGhostStates:
        #    print ghostState
        score = successorGameState.getScore()
        if newPos==currentGameState.getPacmanPosition():
            score -= 100
        ghosts = []

        #print myScore
        #wait wait, you win!
        score += (100+len(foodCoordinates))/(len(foodCoordinates)*9+1)
        if successorGameState.isWin():
            return infinity()

        #get a list of the ghosts positions
        ##for state in newGhostStates:
            #ghosts.append(state.getPosition())

        #print ghosts
       # nearestGhost = infinity(),None
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            ghostDistance = util.manhattanDistance(ghostPos,newPos)
            #if(ghostDistance<nearestGhost[0]):
            #       nearestGhost = ghost
            #pacman doesn't need to run away from ghosts when he has capsule power
            if(ghost.scaredTimer<1):
                score += ghostDistance


        nearest = infinity()
       # foodList = []
        #nevermind, the coordinate might disappear after pacman eats the pellet
        for food in foodCoordinates:
            foodDistance = util.manhattanDistance(food, newPos)
          #  print foodDistance
          #  if (foodDistance < nearest):
             #   nearest = foodDistance
            nearest = min(foodDistance,nearest)
                #reward
        #give pacman urgency to find pellets
        score -=nearest


        return score

        """
        
        
  
            
         def max-value(state, alpha, beta):
            v = -infinity
            for successor in state:
                v = max(v,value(successor,alpha,beta)
                if v>= beta:
                    return v
                a = max(a,v)
                return v    
         def min-value(state, alpha, beta):
            v = infinity
            for successor in state:
                v = min(v,value(successor,alpha,beta)
                if v<= alpha:
                    return v
                a = min(a,v)
                return v    
        """



        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

def infinity():
    return float("inf")

def argmin(seq, fn):
        """Return an element with lowest fn(seq[i]) score; tie goes to first one.
        >>> argmin(['one', 'to', 'three'], len)
        'to'
        """
        best = seq[0];
        best_score = fn(best)
        for x in seq:
            x_score = fn(x)
            if x_score < best_score:
                best, best_score = x, x_score
        return best

def argmax(seq, fn):
        """Return an element with highest fn(seq[i]) score; tie goes to first one.
        >>> argmax(['one', 'to', 'three'], len)
        'three'
        """
        return argmin(seq, lambda x: -fn(x))


def minimax(self,gameState,depth,max=True):
    if depth==0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

    if max:
        bestValue = -(infinity())
        children = gameState.getLegalActions()
        for child in children:
            action = gameState.generateSuccessor(self.index,child)
            v = minimax(action,depth-1,False)
            bestValue = max(bestValue,v)
        return bestValue




class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def max_value(self, gameState, depth):
        terminalState = depth == 0 or gameState.isWin() or gameState.isLose()
        if terminalState:
            return self.evaluationFunction(gameState), Directions.STOP

        plays = gameState.getLegalActions(0)
        v = -infinity(),Directions.STOP
        for ply in plays:
            temp = self.min_value(gameState.generateSuccessor(self.index, ply), 1, depth)[0]
            u,a = v
            if temp > u:
                v = (temp,ply)


        return v

    def min_value(self, gameState, agentIndex, depth):
        lastGhost = gameState.getNumAgents() - 1
        terminalState = depth == 0 or gameState.isWin() or gameState.isLose()
        if terminalState:
            return self.evaluationFunction(gameState), Directions.STOP
        plays = gameState.getLegalActions(agentIndex)  # get legal actions.
        v = infinity(),Directions.STOP
        #woah there, lets not get out of bounds
        if (agentIndex == lastGhost):
            for ply in plays:
                temp = self.max_value(gameState.generateSuccessor(agentIndex, ply), (depth - 1))[0]
                u,a = v
                if temp < u:
                    v = (temp,ply)
        else:
            for ply in plays:
                temp = self.min_value(gameState.generateSuccessor(agentIndex, ply), agentIndex + 1, depth)[0]
                u,a = v
                if temp < u:
                    v = (temp,ply)

        return v

    def getAction(self, gameState):
        return self.max_value(gameState, self.depth)[1]





    # print max_value(gameState, self.depth)



        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, gameState, depth,alpha,beta):
        terminalState = depth == 0 or gameState.isWin() or gameState.isLose()
        if terminalState:
            return self.evaluationFunction(gameState), Directions.STOP

        plays = gameState.getLegalActions(0)
        v = -infinity(),Directions.STOP
        for ply in plays:
            temp = self.min_value(gameState.generateSuccessor(self.index, ply), 1, depth,alpha,beta)[0]
            u,a = v
            if temp > u:
                v = (temp,ply)
            if temp > beta:
                return v
            alpha= max(alpha,temp)


        return v

    def min_value(self, gameState, agentIndex, depth,alpha,beta):
        lastGhost = gameState.getNumAgents() - 1
        terminalState = depth == 0 or gameState.isWin() or gameState.isLose()
        if terminalState:
            return self.evaluationFunction(gameState), Directions.STOP
        plays = gameState.getLegalActions(agentIndex)  # get legal actions.
        v = infinity(),Directions.STOP
        #woah there, lets not get out of bounds
        if (agentIndex == lastGhost):
            for ply in plays:
                temp = self.max_value(gameState.generateSuccessor(agentIndex, ply), (depth - 1),alpha,beta)[0]
                u,a = v
                if temp < u:
                    v = (temp,ply)
                if temp<alpha:
                    return v
                beta = min(beta,temp)
        else:
            for ply in plays:
                temp = self.min_value(gameState.generateSuccessor(agentIndex, ply), agentIndex + 1, depth,alpha,beta)[0]
                u,a = v
                if temp < u:
                    v = (temp,ply)
                if temp < alpha:
                    return v
                beta = min(beta, temp)

        return v

    def getAction(self, gameState):

        alpha = -infinity()
        beta = infinity()
        return self.max_value(gameState, self.depth,alpha,beta)[1]
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    #probability e.g 1/4 * cost e.g 20 mins

    """
    expectimax is weak to adversarial, but outperforms minimax vs random acions. in both cases minimax will win, however.
    values represent average-case outcomes, not worst-case outcomes
    def max-value is max value
    def exp-value
    v = 0
    def value if terminal state tc
    if next agent is max
    return max
    if next agent is exp rt exp
    for each successor of state:
        p = probability(successor)
        v += p*value(successor)
        return v
      Your expectimax agent (question 4)
    """

    def max_value(self, gameState, depth):
        terminalState = depth==0 or gameState.isWin() or gameState.isLose()
        if terminalState:
            return self.evaluationFunction(gameState), Directions.STOP

        plays = gameState.getLegalActions(0)
        v = -infinity(),Directions.STOP
        for ply in plays:
            temp = self.exp_value(gameState.generateSuccessor(self.index, ply), 1, depth)[0]
            u,a = v
            if temp > u:
                v = (temp,ply)
        return v

    def exp_value(self,gameState,agentIndex,depth):
        lastGhost = gameState.getNumAgents() - 1
        terminalState = depth == 0 or gameState.isWin() or gameState.isLose()
        if terminalState:
            return self.evaluationFunction(gameState), Directions.STOP
        plays = gameState.getLegalActions(agentIndex)  # get legal actions.
        p =0
        v = infinity(), Directions.STOP
        # woah there, lets not get out of bounds
        if (agentIndex == lastGhost):
            for ply in plays:
                temp = self.max_value(gameState.generateSuccessor(agentIndex, ply), (depth - 1))[0]
                u, a = v
                #the ghosts likely have the same odds for every direction
                p += 1.0/len(plays) * temp
                v = (p, ply)
        else:
            for ply in plays:
                temp = self.exp_value(gameState.generateSuccessor(agentIndex, ply), agentIndex + 1, depth)[0]
                u, a = v
                # the ghosts likely have the same odds for every direction
                p += 1.0 / len(plays) * temp
                v = (p,ply)
        return v

#    def expectiminimax(self,gameState,depth):

    def getAction(self, gameState):
        return self.max_value(gameState, self.depth)[1]
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      I feel that my original eval function already was very state focused, so this one is quite similar..

        goals we want to teach pacman, in levels of priority:
        #1.) DONT LOSE (dont die)
        #2.) WIN (eat ALL pellets)
`       #3.) capsule collection is largely an afterthough. I do not want to promote this goal unless it is confirmed to be safe
      DESCRIPTION: <write something here so we know what you did>
    """
    currentCapsules = currentGameState.getCapsules()
    foodCoordinates = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    score += (2000 + len(foodCoordinates)) / (len(foodCoordinates) * 9 + 1)

    nearestGhost = 100
    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        ghostDistance = util.manhattanDistance(ghostPos, newPos)
        # if(ghostDistance<nearestGhost[0]):
        #       nearestGhost = ghost
        # pacman doesn't need to run away from ghosts when he has capsule power
        nearestGhost = min(nearestGhost,ghostDistance)
        if (ghost.scaredTimer < 1):
            score += ghostDistance*100
        elif ghostDistance <2:
            score += 1000

    #for capsule in currentCapsules:
     #   capsuleDist = util.manhattanDistance(capsule, newPos)
      #  if nearestGhost > 3:
       #     score += (100 - capsuleDist)



    nearest = infinity()
    # foodList = []
    # nevermind, the coordinate might disappear after pacman eats the pellet
    for food in foodCoordinates:
        foodDistance = util.manhattanDistance(food, newPos)
        #  print foodDistance
        #  if (foodDistance < nearest):
        #   nearest = foodDistance
        nearest = min(foodDistance, nearest)

        score -= nearest
    if currentGameState.isWin():
        return infinity()
    if currentGameState.isLose():
        return -infinity()
    return score
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

