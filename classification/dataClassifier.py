# dataClassifier.py
# -----------------
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


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import util
from pacman import GameState

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

##Code taken from https://github.com/hyzhangsf/cs188_proj5/blob/master/dataClassifier.py, fully understood but for gods sake writing thing this myself had my struggling with the annoyingly
##designed framework (how even did they think Queue's should work) for so long it wasn't worth it

def getPartitionNum(pixels):
    '''
    :param pixels: a 2-dim array of pixels representing an image
    :return: number of regions of  the image
    '''
    from collections import Counter
    def pointInImage(x, y):
        return (0 <= x < DIGIT_DATUM_HEIGHT) and (0 <= y < DIGIT_DATUM_WIDTH)

    def neighbours(x, y):
        '''
        :param x: x coordinate of point
        :param y: y coordinate of point
        :return: a set of numbering points of point (x, y)
        '''
        candidates_neighbours = ((x, y-1), (x, y+1), (x+1, y), (x-1, y), (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1))
        return {(x, y) for x, y in candidates_neighbours if pointInImage(x, y)}

    def unexplored_neighbours(x, y, explored_set):
        '''
        :param x: x coordinate of point
        :param y: y coordinate of point
        :param explored_set: a set containing explored points
        :return: a set of non-explored sets
        '''
        return neighbours(x, y) - explored_set

    def pointOnEdge(x, y):
        return pixels[x][y] > 0

    def pointNotOnEdge(x, y):
        return not pixels[x][y] > 0

    def bfs(explored_set, start_point, spaceIndex, partition):
        # use bfs to color the image
        queue = [start_point]
        showEnqueue = pointOnEdge if pointOnEdge(*start_point) else pointNotOnEdge
        while len(queue) > 0:
            x, y = current = queue.pop(0)
            if ((x, y) not in explored_set) and showEnqueue(x, y):
                partition[current] = spaceIndex
                explored_set.add(current)
                queue.extend(unexplored_neighbours(x, y, explored_set))
        return partition

    def partitionIsComplete(partition):
        for x in xrange(DIGIT_DATUM_HEIGHT):
            for y in xrange(DIGIT_DATUM_WIDTH):
                if (x, y) not in partition:
                    return False, (x, y)
        return True, None

    exploredPoints = set()
    partition = {}
    spaceIndex = 0
    isComplete, startPoint = partitionIsComplete(partition)
    while not isComplete:
        partition = bfs(exploredPoints, startPoint, spaceIndex, partition)
        spaceIndex += 1
        isComplete, startPoint = partitionIsComplete(partition)
    c = Counter(partition.values()).items()
    return len(filter(lambda t: t[1] > 3, c))

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.
    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).
    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
    1. Feature 'upper':
        'upper' = 1 if upper half has more points
    2. Feature 'left':
        'left' = 1 if left half has more points
    3. Feature "HorizontalSymmetry":
        'hs' = 1 if left and right side are symmetry (30% points coincide)
    4. Feature "regions":
        'regions{i}' = 1 if there are exactly i connected regions in image, else 0
    5. Feature "empty":
        'empty{i}' = 0 if line i is white, else 1
    6. Feature "hole":
        'hole{i}' == 1 if line i has a hole of while pixles else 0
    ##
    """
    features =  basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    # adds a feature, multiple times to give it more weight in the final decision
    def addFeature(name, value, time):
        features[name] = value
        for i in xrange(time):
            features[str(i) + '--' + name] = features[name]

    TOTAL_PIXELS = DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT

    # count all the pixels in the respective parts of the picture
    pixels = datum.getPixels()
    upper = sum([pixels[row][col] > 0 for row in xrange(DIGIT_DATUM_HEIGHT / 2)
                 for col in xrange(DIGIT_DATUM_WIDTH)])
    lower = sum([pixels[row][col] > 0 for row in xrange(DIGIT_DATUM_HEIGHT / 2, DIGIT_DATUM_HEIGHT)
                 for col in xrange(DIGIT_DATUM_WIDTH)])
    left = sum([pixels[row][col] > 0 for row in xrange(DIGIT_DATUM_WIDTH)
                for col in xrange(DIGIT_DATUM_WIDTH / 2)])
    right = sum([pixels[row][col] > 0 for row in xrange(DIGIT_DATUM_WIDTH)
                 for col in xrange(DIGIT_DATUM_WIDTH / 2, DIGIT_DATUM_WIDTH)])

    # add features compared to what halves have more white pixels
    addFeature('left', left > right, 2)
    addFeature('upper', upper > lower, 2)

    # count the connected regions in the picture
    connectedRegionsCount = getPartitionNum(pixels)
    for i in xrange(1, 5):
        # add a '1' when it's that amount of regions in the picture, drawback, no more than 5 connected regions could be counted
        addFeature('regions'+str(i), connectedRegionsCount == i, 6)

    # check full vertical lines
    for y in range(DIGIT_DATUM_HEIGHT):
        # check if line is empty
        pixelsInbinary = [bool(datum.getPixel(x, y)) for x in xrange(DIGIT_DATUM_WIDTH)]
        numBlackPixels = sum(pixelsInbinary)
        addFeature('empty'+str(y), numBlackPixels > 0, 3)

        # check if there's a hole inbetween the sides
        leftEdge = ((DIGIT_DATUM_WIDTH-1)-pixelsInbinary[::-1].index(True)) if numBlackPixels else 0
        rightEdge = pixelsInbinary.index(True) if numBlackPixels else 0
        width = leftEdge - rightEdge
        addFeature('hole'+str(y), width + (width > 1) > numBlackPixels, 2)

        # check horizontal symmetry (more than 30% of pixels the same = true)
        # hs: horizontal symmetrical
        hs = sum([pixels[x][y] > 0 and pixels[DIGIT_DATUM_WIDTH - 1 - x][y] > 0 for x in xrange(DIGIT_DATUM_WIDTH / 2)])
        addFeature('hs' + str(y), hs > 0.3*TOTAL_PIXELS/2, 2)
    return features

def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    nextState = state.generateSuccessor(0, action) #generate the state pacman is in after he has taken the action
    pacPos = nextState.getPacmanPosition() #Pacman's position
    ghostPos = nextState.getGhostPositions() #List of ghost positions

    features['score'] = nextState.getScore() #The score assigned to a state

    minGhostDistance = None 
    for ghost in ghostPos: #for each position of a ghost
        ghostDistance = util.manhattanDistance(pacPos, ghost) #get the distance between pacman and the ghost
        if minGhostDistance == None or ghostDistance < minGhostDistance: #store the shortest distance between pacman and a ghost
            minGhostDistance = ghostDistance
    features['ghostDistance'] = minGhostDistance if minGhostDistance != None else 0 #use 0 as distance if there are no ghosts

    #the same functionality as for ghosts, but now for food
    minFoodDistance = None
    for foodPos in nextState.getFood().asList(): 
        foodDistance = util.manhattanDistance(pacPos, foodPos)
        if minFoodDistance == None or foodDistance < minFoodDistance:
            minFoodDistance = foodDistance
    features['foodDistance'] = minFoodDistance if minFoodDistance != None else 0

    features['foodLeft'] = nextState.getNumFood() #The amount of food pellets left in the game

    return features


def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basicFeatureExtractorDigit(datum)
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    #     prediction = guesses[i]
    #     truth = testLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print rawTestData[i]
    #         break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    if not options.classifier == 'minicontest':
        print "using enhanced features?:\t" + str(options.features)
    else:
        print "using minicontest feature extractor"
    print "training set size:\t" + str(options.training)
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif(options.data=="pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print "using automatic tuning for naivebayes"
            classifier.automaticTuning = True
        else:
            print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print "using automatic tuning for MIRA"
            classifier.automaticTuning = True
        else:
            print "using default C=0.001 for MIRA"
    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    if(options.data=="pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print string3
        printImage(features_odds)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
