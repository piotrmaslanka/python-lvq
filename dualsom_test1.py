from __future__ import division
# The LVQ Project
# -------------- imports
import numpy as np
from lvq import LVQNeuralNet, trainlvq, \
                trainsom, differentiate_voronoi, \
                assess2, assess, ClippingAreaGenerator, \
                normalize_each_dimension_separately, \
                differentiate_knearest, \
                trainsom_equipresent, \
                trainlvq_equipresent, make_a_network, \
                knn_eliminate, classify_knn
from copy import deepcopy as copy
from math import e
# -------------- definitions


# -------------- code

# Read input data   -    PARKINSONS.DATA SPECIFIC
data = []
labels = []
with open('parkinsons.data', 'r') as f:
    f.next()        # skip header
    for row in f:
        x = map(float, row.replace('\n','').split(',')[1:])

        labels.append(x[16])    # x[16] is status
        del x[16]
        data.append(x)

data = np.array(data)   # Only data, labels have been separated out earlier


# --------------- perform kNN elimination
#data, labels, premoved = knn_eliminate(data, labels, 1)

# ----- general LVQ code

data = normalize_each_dimension_separately(data)

# Find how many unique labels are there, and what are they
unique_labels = set(labels)
how_many_unique_labels = len(unique_labels)

def theta_maker(max_s):
    def theta(v1, v2, s):
        ln = np.linalg.norm(v1-v2)
        return pow(e, -pow(ln, 2)/16) * (max_s-s)/max_s
    return theta

cag = ClippingAreaGenerator(data)

healthy = [vec for vec, lab in zip(data, labels) if lab == 0]
ill = [vec for vec, lab in zip(data, labels) if lab == 1]

for h_neu in xrange(1, 30):
    for i_neu in xrange(1, 60):
        results = []
        for attempt in xrange(0, 5):

            nh = LVQNeuralNet(len(data[0]))
            ni = LVQNeuralNet(len(data[0]))
            make_a_network(nh, h_neu, cag)
            make_a_network(ni, i_neu, cag)

            trainsom(nh, healthy, 0.1, 30, theta_maker(30))
            trainsom(ni, ill, 0.1, 30, theta_maker(30))

            for neuron in nh.neurons: neuron.label = 0
            for neuron in ni.neurons: neuron.label = 1

            lvq = nh + ni

            trainlvq(lvq, data, labels, 0.1, 30)

            results.append(assess2(lvq, data, labels))

        A1 = [x[0] for x in results]
        A2 = [x[1] for x in results]
#        A1.remove(max(A1))
#        A1.remove(min(A1))
#        A2.remove(max(A2))
#        A2.remove(min(A2))                        

        print '%s %s %s %s' % (h_neu, i_neu, sum(A1)/len(A1), sum(A2)/len(A2))