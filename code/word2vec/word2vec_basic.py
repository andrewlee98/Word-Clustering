# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn import metrics

import matplotlib.pyplot as plt


filename = 'corpus.txt'

# cwd = os.getcwd()
# filename = cwd + filename

def flatten(l):
    try:
        return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if \
        type(l) is list else [l]
    except IndexError:
        return []

# Read the data into a list of strings.
def read_data(filename):
    with open(filename) as f:
        data = [line.rstrip('\n') for line in open(filename)]
    data = [word_tokenize(datum) for datum in data]

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    tokenized_docs_no_punctuation = []
    for review in data:
        new_review = []
        for token in review:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)
        tokenized_docs_no_punctuation.append(new_review)

    # stem
    porter = PorterStemmer()
    docs_stemmed = []
    for doc in tokenized_docs_no_punctuation:
        final_doc = []
        for word in doc:
            final_doc.append(porter.stem(word))
        docs_stemmed.append(final_doc)

    # flatten into one dimensional list
    data = flatten(docs_stemmed)
    data = [x.lower() for x in data]

    data = [word for word in data if word not in stopwords.words('english')]


    # clean digits
    for item in data[:]:
        if any(char.isdigit() for char in item) or len(item)<2:
            data.remove(item)

    return data


words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
# vocabulary_size = 50000
vocabulary_size = 1000

def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = \
build_dataset(words, vocabulary_size)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 50001 # 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# write embeddings to file
with open('embeddings.txt','w+') as f:
    for vec, word in zip(final_embeddings, reverse_dictionary):
        f.write(reverse_dictionary[word] + ": " + \
        str(["{0:0.2f}".format(i) for i in vec]) + "\n\n")

def get_centroids(vecs, labels):
    centroids = []
    temp_sum = [0] * len(vecs[0])
    for i in range(0, max(labels)+1):
        for label,vec in zip(labels, vecs):
            if label == i:
                temp_sum = [sum(x) for x in zip(temp_sum, vec)]
        temp_sum = [x / (labels == i).sum() for x in temp_sum]
        centroids.append(temp_sum)
    return centroids
    

def intra_cluster_dist(X, labels, cluster, num_items_in_cluster, centroid):
    total_dist = 0
    #for every item in cluster j, compute the distance the the center of cluster j, take average
    for k in range(num_items_in_cluster):
        dist = np.linalg.norm(X[labels==cluster]-centroid)
        total_dist = dist + total_dist
    return total_dist/num_items_in_cluster

def davies_bouldin(X, labels, cluster_ctr):
    #get the cluster assignemnts
    clusters = list(set(labels))
    #get the number of clusters
    num_clusters = len(clusters)
    #array to hold the number of items for each cluster, indexed by cluster number
    num_items_in_clusters = [0]*num_clusters
    #get the number of items for each cluster
    for i in range(len(labels)):
        num_items_in_clusters[labels[i]] += 1
    max_num = -9999
    for i in range(num_clusters):
        s_i = intra_cluster_dist(X, labels, clusters[i], num_items_in_clusters[i], cluster_ctr[i])
    for j in range(num_clusters):
        if(i != j):
            s_j = intra_cluster_dist(X, labels, clusters[j], num_items_in_clusters[j], cluster_ctr[j])
            m_ij = np.linalg.norm([a - b for a,b in zip(cluster_ctr[clusters[i]],cluster_ctr[clusters[j]])])
            r_ij = (s_i + s_j)/m_ij
            if(r_ij > max_num):
                max_num = r_ij
    return max_num

def dbscan(vecs, disp):
    # normalize embeddings
    vecs = StandardScaler().fit_transform(vecs)

    db = cluster.DBSCAN(eps=.3, min_samples=10).fit(vecs)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    if n_clusters_ < 1:
        n_clusters_ = n_clusters_ + 1
    cluster_lists = [[] for x in xrange(n_clusters_)]
    with open("clusters.txt", "w+") as text_file:
        text_file.write("=====DBSCAN (" + str(n_clusters_) + " clusters) :=====\n")
        for x in range(0,n_clusters_):
            for word_num in range(0, len(labels)):
                if labels[word_num] == x-1:
                    cluster_lists[x].append(reverse_dictionary[word_num])
            # write clusters to text file
            text_file.write("cluster " + str(x) + ": " + str(cluster_lists[x]) + \
            "\n\n")
        text_file.write("\n\n\n")

    if max(labels) < 2:
        return

    silhouette = metrics.silhouette_score(vecs, y_pred, metric = 'sqeuclidean')
    centroids = get_centroids(vecs, y_pred)
    davies = davies_bouldin(vecs, y_pred, centroids)

    if disp:
        print("dbscan:")
        print("silhouette score: " + silhouette)
        print("Davies-Bouldin Index: " + str(davies_bouldin(vecs, labels ,centroids)))
        print("\n")



def spectral(vecs, n_clusters, disp):
    spectral = cluster.SpectralClustering(n_clusters,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    spectral.fit(vecs);
    y_pred = spectral.labels_.astype(np.int)

    if disp:
        # create lists of clustered words
        cluster_lists = [[] for x in xrange(n_clusters)]
        with open("clusters.txt", "a") as text_file:
            text_file.write("=====spectral:=====\n")
            for x in range(0,n_clusters):
                for word_num in range(0, len(y_pred)):
                    if y_pred[word_num] == x:
                        cluster_lists[x].append(reverse_dictionary[word_num])
                # write clusters to text file
                text_file.write("cluster #" + str(x + 1) + ": " + \
                str(cluster_lists[x]) + "\n\n")
            text_file.write("\n\n\n")

    silhouette = metrics.silhouette_score(vecs, y_pred, metric = 'sqeuclidean')
    centroids = get_centroids(vecs, y_pred)
    davies = davies_bouldin(vecs, y_pred, centroids)

    if disp:
        print("spectral:")
        print("silhouette score: " + str(silhouette))
        print("Davies-Bouldin Index: " + str(davies_bouldin(vecs, y_pred, centroids)))
        print("\n")

    return davies, silhouette

def kmeans(vecs, n_clusters, disp):
    kmeans = cluster.KMeans(n_clusters, init='k-means++').fit(vecs)
    y_pred = kmeans.labels_.astype(np.int)

    if disp:
        # create lists of clustered words
        cluster_lists = [[] for x in xrange(n_clusters)]
        with open("clusters.txt", "a") as text_file:
            text_file.write("=====k-means:=====\n")
            for x in range(0,n_clusters):
                for word_num in range(0, len(y_pred)):
                    if y_pred[word_num] == x:
                        cluster_lists[x].append(reverse_dictionary[word_num])
                # write clusters to text file
                text_file.write("cluster #" + str(x + 1) + ": " + \
                str(cluster_lists[x]) + "\n\n")
            text_file.write("\n\n\n")

        silhouette = metrics.silhouette_score(vecs, y_pred, metric = 'sqeuclidean')
        centroids = get_centroids(vecs, y_pred)
        davies = davies_bouldin(vecs, y_pred, centroids)

    if disp:
        print("kmeans:")
        print("silhouette score: " + str(silhouette))
        print("Davies-Bouldin Index: " + str(davies_bouldin(vecs, y_pred, centroids)))
        print("\n")

    return davies, silhouette

dbscan(final_embeddings, False)

# plot for spectral
n_clusters = []
sils = []
davs = []
for i in range(2,101):
    n_clusters.append(i)
    davies, silhouette = spectral(final_embeddings, i, False)
    sils.append(silhouette)
    davs.append(davies)

def find_optimal_clusters(n_clusters, davs, sils):
    p0 = (n_clusters[0], davs[0])
    p1 = (n_clusters[len(n_clusters) - 1], davs[len(davs) - 1])

    a = p0[1] - p1[1]
    b = p1[0] - p0[0]
    c = (p0[0] - p1[0]) * p0[1] + (p1[1] - p0[1]) * p0[0]

    max_dist = 0
    for x, y in zip(n_clusters, davs):
        d = abs(a*x + b*y + c) / math.sqrt(a**2 + b**2)
        if d > max_dist:
            max_dist = d
            elbow = (x,y)
    return elbow

def plot_metrics(n_clusts, dav, sil, graph_title, elbow):
    plt.figure()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(elbow[0], elbow[1], 'gs')
    ax1.plot(n_clusts, dav, 'r--')
    ax2.plot(n_clusts, sil, 'b--')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Davies-Bouldin Index', color='r')
    ax2.set_ylabel('Silhouette Score', color='b')
    plt.title('Metrics for ' + graph_title + ' Clustering')
    plt.savefig(graph_title + '.png', bbox_inches='tight')
    print(graph_title + ": " elbow)

elbow = find_optimal_clusters(n_clusters, davs, sils)
plot_metrics(n_clusters, davs, sils, 'Spectral', elbow)

# plot for K-Means
n_clusters = []
sils = []
davs = []
for i in range(2,101):
    n_clusters.append(i)
    davies, silhouette = spectral(final_embeddings, i, False)
    sils.append(silhouette)
    davs.append(davies)

elbow = find_optimal_clusters(n_clusters, davs, sils)
plot_metrics(n_clusters, davs, sils, 'K-Means', elbow)

# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy.")
