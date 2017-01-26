# mira.py
# -------
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


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA. Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.

        This code can be found on: https://github.com/anthony-niklas/cs188/blob/master/p5/mira.py
        Credits go to the respective owners of the github project.
        """
        "*** YOUR CODE HERE ***"
        bestWeights = []
        bestAccuracy = None
        for c in Cgrid:
            weights = self.weights.copy() #copy it so we don't already edit the self.weights list
            for n in range(self.max_iterations): #we may not exceed the amount of max iterations
                for i, datum in enumerate(trainingData): #keep track of a counter as well as the current data we're looking at in trainingData
                    bestScore = None
                    bestY = None
                    for y in self.legalLabels: #for each posible move check
                        score = datum * weights[y] #calculate the score by using the weight
                        if score > bestScore or bestScore == None: #if we've found a better score
                            bestScore = score #update the new best score
                            bestY = y #update it's output value

                    actualY = trainingLabels[i] #the output value we were supposed to find
                    if bestY != actualY: #if we've found something else than the actualY 
                        f = datum.copy() #copy the data so we don't actually edit the real data
                        tau = min(c, ((weights[bestY] - weights[actualY]) * f + 1.0) / (2.0 * (f * f))) #use the forumla given on the website
                        f.divideAll(1.0 / tau)
                        
                        weights[actualY] = weights[actualY] + f #update the weights
                        weights[bestY] = weights[bestY] - f
            
            # Check the accuracy associated with this c
            correct = 0
            guesses = self.classify(validationData) #the already written classify function returns the guesses
            for i, guess in enumerate(guesses): #for each guess
                correct += (validationLabels[i] == guess and 1.0 or 0.0) #count the number of correct guesses
            accuracy = correct / len(guesses) #accuracy is the same as the amount of correct guesses divided by the amount of total guesses
            
            if accuracy > bestAccuracy or bestAccuracy is None: #If we've found a better accuracy
                bestAccuracy = accuracy #use this accuracy as the new best accuracy
                bestWeights = weights #use these weights as the new best weights
    
        self.weights = bestWeights #set the weights list to the best found weights

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


