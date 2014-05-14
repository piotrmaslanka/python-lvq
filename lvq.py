from __future__ import division
import numpy as np
from collections import defaultdict
from random import random, choice

rnd = lambda: (random()-0.5) * 2   # return a random number from [-1; 1]

get_fel = lambda x: x[0]        # getter: first element
get_sel = lambda x: x[1]        # setter: second element

def histogramize(data):
    """Converts a sequence to a dict(item => amount of these items in list)"""
    return dict(((label, data.count(label)) for label in set(data)))

class ClippingAreaGenerator(object):
    """An intelligent random vector generator (see 4.2)"""
    def __init__(self, data):
        self.n = len(data[0])
        kv = sum(data) / len(data)
        self.maxs = [max([x[i] for x in data])/4-kv[i] for i in xrange(0, self.n)]
        self.mins = [min([x[i] for x in data])/4-kv[i] for i in xrange(0, self.n)]
        # so numbers must fall in range (mins..maxs) for given dimension i

    def get_vector(self):
        s = []
        for i in xrange(0, self.n):
            q = random()*(self.maxs[i] - self.mins[i]) - self.mins[i]
            s.append(q)
        return np.array(s)

class LVQNeuron(object):
    """Class that represents a single neuron"""
    def __init__(self, n, label):
        """@param n: Generator vector
        @param label: Class label"""
        self.vector = np.array([rnd() for x in xrange(0, n)])
        self.label = label

    def dist(self, vector):
        q =  np.linalg.norm(self.vector - vector)
        return q

    def train(self, vector, label, alpha):
        if label == self.label:
            self.vector += (vector - self.vector) * alpha
        else:
            self.vector -= (vector - self.vector) * alpha
            
class LVQNeuralNet(object):
    def __init__(self, n):        
        self.neurons = []
        self.n = n

    def add_random_neurons(self, k, label):
        """Adds k neurons of label class with randomly generated vectors"""
        self.neurons.extend([LVQNeuron(self.n, label) for x in xrange(0, k)])

    def get_classifier(self, vector):
        """Returns the classifier neuron"""
        s = [(neuron, neuron.dist(vector)) for neuron in self.neurons]
        q = min(s, key=get_sel)[0]
        return q

    def classify(self, vector):
        return self.get_classifier(vector).label

    def __add__(self, on):
        assert on.n == self.n
        s = LVQNeuralNet(self.n)
        s.neurons = self.neurons + on.neurons
        return s


def trainlvq(lvq, input_vecs, label_vecs, a, series):
    for x in xrange(0, series):
        for lab, vec in zip(label_vecs, input_vecs):
            lvq.get_classifier(vec).train(vec, lab, a)

        a = a / (1 + a)

def trainlvq_equipresent(lvq, input_vecs, label_vecs, a, series, psp):

    # Prepare source data
    dataclasses = defaultdict(list)        # dict: label => sequence of (vector)
    for label, vec in zip(label_vecs, input_vecs):
        dataclasses[label].append(vec)

    for s in xrange(0, series):
        for _ in xrange(0, psp):
            for label in dataclasses.iterkeys():            
                vec = choice(dataclasses[label])
                lvq.get_classifier(vec).train(vec, label, a)


def trainsom_equipresent(lvq, input_vecs, label_vecs, a, series, theta, psp):
    """@param psp: how many entries from each label to present during a series"""

    # Prepare source data
    dataclasses = defaultdict(list)        # dict: label => sequence of (vector)
    for label, vec in zip(label_vecs, input_vecs):
        dataclasses[label].append(vec)

    for s in xrange(0, series):
        for _ in xrange(0, psp):
            for label in dataclasses.iterkeys():
                vec = choice(dataclasses[label])
                winner = lvq.get_classifier(vec)
                for i in xrange(0, len(lvq.neurons)):
                    lvq.neurons[i].vector += theta(lvq.neurons[i].vector, winner.vector, s) * a * (vec - winner.vector)

def trainsom(lvq, input_vecs, a, series, theta):
    for s in xrange(0, series):
        for vec in input_vecs:
            winner = lvq.get_classifier(vec)
            for i in xrange(0, len(lvq.neurons)):
                lvq.neurons[i].vector += theta(lvq.neurons[i].vector, winner.vector, s) * a * (vec - winner.vector)

def differentiate_voronoi(lvq, input_vecs, label_vecs):
    possible_labels = set(label_vecs)
    # Annotate input vector with a classifier
    annots = [(vec, label, lvq.get_classifier(vec)) for vec, label in zip(input_vecs, label_vecs)]

    # Get a dict: neuron => list of vectors assigned to it
    ndict = [
             (neuron, [(vec, label) for vec, label, neur in annots if neur == neuron])
             for neuron in lvq.neurons
            ]

    # Kill useless neurons
    nds = {}
    for neuron, vectors in ndict:
        if len(vectors) == 0:
            lvq.neurons.remove(neuron)
        else:
            nds[neuron] = vectors

    # Assign a category
    for neuron, vectors in nds.iteritems():
        voting_dict = dict([(label, 0) for label in possible_labels])
        for vec, label in vectors:
            voting_dict[label] += 1

        neuron.label = max(voting_dict.iteritems(), key=get_sel)[0]

    # Check if all labels have been used...
    if len(possible_labels) != len(set([neuron.label for neuron in lvq.neurons])):
        raise Exception, 'Map failed to differentiate properly'


def differentiate_knearest(lvq, input_vecs, label_vecs, k=5):
    labels = set(label_vecs)

    for neuron in lvq.neurons:
        neighbors = [(label, neuron.dist(vec)) for vec, label in zip(input_vecs, label_vecs)]
        neighbors.sort(key=get_sel)

        neighbors = neighbors[:k]

        if len(neighbors) != k:
            raise Exception, 'Cannot pick %s neighbors' % k

        neuron.label = max([(label, neighbors.count(label)) for label in labels],
                           key = get_fel)[0]

    # Check if all labels have been used...
    if len(labels) != len(set([neuron.label for neuron in lvq.neurons])):
        raise Exception, 'Map failed to differentiate properly'

def assess(lvq, input_data, label_data):
    """Assess the network performance.
    @return: (percentage of correctly classified data"""
    correct = 0.0
    incorrect = 0.0
    for lab, vec in zip(label_data, input_data):
        if lvq.classify(vec) == lab:
            correct += 1
        else:
            incorrect += 1

    return 100*correct/(correct+incorrect)


def assess2(lvq, input_data, label_data):
    """Assess the network performance using another performance metric
    @return: dict: label => (percentage of correctly classified data in that label)"""
    labels = set(label_data)
    sample_count = histogramize(label_data)

    correct_for = dict([(label, 0) for label in labels])

    for lab, vec in zip(label_data, input_data):
        if lvq.classify(vec) == lab:
            correct_for[lab] += 1

    for label in labels:
        correct_for[label] = correct_for[label] / sample_count[label] * 100

    return correct_for

def normalize_each_dimension_separately(data):
    """Data is a sequence of vectors. Data will be normalized to
    [-1; 1] so that each dimension is considered separately.

    Current standard routine used to normalize input data"""
    # Create maximum and minimum vector
    maxs = np.array([max([x[i] for x in data]) for i in xrange(0, len(data[0]))])
    mins = np.array([min([x[i] for x in data]) for i in xrange(0, len(data[0]))])

    # Normalize the values to [-1, 1]
    data = [(((vec-mins)/(maxs-mins))-0.5)*2 for vec in data]   # crude mapminmax
    return np.array(data)

def normalize_dimensions_shared(data):
    """Data is sequence of vectors. Max and min is found among numeric entries
    here, and everything is normalized.

    Proved to be detrimential to results, and will not be considered in the future"""
    mx = max([max(x) for x in data])
    mn = min([min(x) for x in data])

    data = [(((vec-mn)/(mx-mn))-0.5)*2 for vec in data]   # crude mapminmax
    return np.array(data)

def make_a_network(lvq, neurons, cag):
    """Makes a network with class-unspecified neurons"""
    lvq.add_random_neurons(neurons, 0.0)

    for neuron in lvq.neurons:
        neuron.vector = cag.get_vector()    
        

def knn_eliminate(data, labels, k):
    """Performs k-NN filtering. Returns a tuple (new data, new labels, points removed)"""
    premoved = 0
    newdata = []
    newlabels = []
    
    apZ = 0
    apJ = 0
    
    for vec, lab in zip(data, labels):
        nd = sorted(zip(data, labels), key=lambda x: np.linalg.norm(vec-x[0]))
        dp = nd[1:k+1]
        if True in [l == lab for d, l in dp]:
            # accept point
            newdata.append(vec)
            newlabels.append(lab)
        else:
            if lab == 0: apZ += 1
            else: apJ += 1
            premoved += 1
            
    print '%s %s' % (apZ, apJ)
    return newdata, newlabels, premoved
    
def classify_knn(data, labels, k):
    correct_for = dict([(label, 0) for label in set(labels)])
    sample_count = histogramize(labels)
    
    correct = 0
    
    for vec, lab in zip(data, labels):
        nd = sorted(zip(data, labels), key=lambda x: np.linalg.norm(vec-x[0]))
        dp = map(get_sel, nd[1:k+1])    # dn is array of labels
        
        histogram = histogramize(dp).items()
        histogram.sort(key=get_sel, reverse=True)
        answer = histogram[0][0]
        
        if answer == lab:
            correct_for[lab] += 1
            correct += 1
        
    for label in set(labels):
        correct_for[label] = correct_for[label] / sample_count[label] * 100

    return correct_for, (correct / len(labels) * 100)        
        
        
    