
# coding: utf-8

# <img src="http://www.clipartkid.com/images/842/big-data-cloud-crowd-sensing-data-mining-distribueret-machine-learning-fy5yap-clipart.png" width="700" height="400"/>

#  # Machine Learning Project: (adapted from CS194-16) Introduction to Data Science csc599.70
#  
#  
# **Name**: *CHERNO S JALLOW*
# 
# **Student ID**: *xxx16852*
# 
# 
# Introduction to Machine Learning: Clustering and Regression
# ===
# 
# ## Overview
# 
# In this assignment, we will use machine learning techniques to perform data analysis and learn models about our data. We will use a real world music dataset from [Last.fm](http://last.fm) for this assignment. There are two parts to this assignment: In the first part we will look at Unsupervised Learning with clustering and in the second part, we will study Supervised Learning. The play data (and user/artist matrix) comes from the [Last.fm 1K Users dataset](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html), while the tags come from [the Last.fm Music Tags dataset](http://musicmachinery.com/2010/11/10/lastfm-artisttags2007/). You won't have to interact with these datasets directly, because we've already preprocessed them for you.

# ### Introduction to Machine Learning
# 
# Machine learning is a branch of artifical intelligence where we try to find hidden structure within data. 
# For example, lets say you are hired as a data scientist at a cool new music playing startup. You are given access to 
# logs from the product and are asked find out what kinds of music are played on your website and how you can promote songs that will be 
# popular. In this case we wish to extract some structure from the raw data we have using machine learning.
# 
# There are two main kinds of machine learning algorithms:
# 
# 1. Unsupervised Learning - is the branch where we don't have any ground truth (or labeled data) that can help our training process. There are many approaches to unsupervised learning which includes topics like Clustering,  Mixture Models, Hidden Markov Models etc. In this assignment we will predominantly look at clustering.
# 2. Supervised Learning - we have training data which is labeled (either manually or from historical data) and we try to make predictions about those labels on new, unlabeled, data. There are similarly several approaches to supervised learning - various classification and regression techniques all the way up to Support Vector Machines and Convolutional Neural Networks. In this assignment we'll explore two regression algorithms - Least Squares Linear Regression and Regression Trees. 
# 
# Many of the techniques you'll be using (like testing on a validation set) are of critical importance to the modeling process, regardless of the technique you're using, so keep these in mind in your future modeling efforts.

# ### Application
# 
# Your assignment is to use machine learning algorithms for two tasks on a real world music dataset from Last.fm. The goal in the first part is to cluster artists and try to discover all artists that belong a certain genre. In the second part, we'll use the same dataset and attempt to predict how popular a song will be based on a number of features of that song. One component will involve incorporating cluster information into the models.

# ### Files
# 
# Data files for this assignment can be found via this link. Either download directly to your VM, or to your host and use drag-and-drop. Create a directory ~/HWs/HW3 and put the archive file there. Unpack it with <code>tar xvzf file</code>
# 
# The zip file includes the following files:
# 
# * **artists-tags.txt**, User-defined tags for top artists
# * **userart-mat-training.csv**, Training data containing a matrix mapping artist-id to users who have played songs by the artists
# * **userart-mat-test.csv**, Test data containing a matrix mapping artist-id to users who have played songs by the artists
# * **train_model_data.csv**, Aggregate statsitstics and features about songs we'll use to train regression models.
# * **validation_model_data.csv**, Similar statistics computed on a hold-out set of users and songs that we'll use to validate our regression models.
# 
# We will explain the datasets and how they need to used in the assignment sections.

# ### Deliverables
# 
# Complete the all the exercises below and turn in a write up in the form of an IPython notebook, that is, **an .ipynb file**.
# The write up should include your code, answers to exercise questions, and plots of results.
# This time you will submit it directly to bCourses. 
# 
# We recommend that you do your work in a **copy of this notebook**, in case there are changes that need to be made that are pushed out via github. In this notebook, we provide code templates for many of the exercises. They are intended to help with code re-use, since the exercises build on each other, and are highly recommended. Don't forget to include answers to questions that ask for natural language responses, i.e., in English, not code!

# ### Guidelines
# 
# #### Code
# 
# This assignment can be done with basic python, matplotlib and scikit-learn.
# Feel free to use Pandas, too, which you may find well suited to several exercises.
# As for other libraries, please check with course staff whether they're allowed.
# 
# You're not required to do your coding in IPython, so feel free to use your favorite editor or IDE.
# But when you're done, remember to put your code into a notebook for your write up.
# 
# #### Collaboration
# 
# This assignment is to be done individually.  Everyone should be getting a hands on experience in this course.  You are free to discuss course material with fellow students, and we encourage you to use Internet resources to aid your understanding, but the work you turn in, including all code and answers, must be your own work.

# ## Part 0: Preliminaries
# 
# ### Exercise 0
# 
# Read in the file **artists-tags.txt** and store the contents in a DataFrame. The file format for this file is `ArtistID|ArtistName|Tag|Count`. The fields mean the following:
# 
# 1. ArtistID : a unique id for an artist (Formatted as a [MusicBrainz Identifier](https://musicbrainz.org/doc/MusicBrainz_Identifier))
# 2. ArtistName: name of the artist
# 3. Tag: user-defined tag for the artist
# 4. Count: number of times the tag was applied
# 
# Similarly, read in the file **userart-mat-training.csv** . The file format for this file is `ArtistID, user_000001, user_000002, .... user_001000`. i.e. There are 846 such columns in this file and each column has a value 1 if the particular user played a song from this artist.

# In[54]:

from pylab import *
get_ipython().magic('matplotlib inline')
import pandas as pd

DATA_PATH = "HW3" # Make this the /path/to/the/data

def parse_artists_tags(filename):
    df = pd.read_csv(filename, sep="|", names=["ArtistID", "ArtistName", "Tag", "Count"])
    return df

def parse_user_artists_matrix(filename):
    df = pd.read_csv(filename)
    return df

artists_tags = parse_artists_tags(DATA_PATH + "/artists-tags.txt")
user_art_mat = parse_user_artists_matrix(DATA_PATH + "/userart-mat-training.csv")

print ("Number of tags %d" % artists_tags.Tag.count() ) # Change this line after calculating the number of tags. 
print ("Number of artists %d" % artists_tags.ArtistID.count()) # Change this line. 
print ("Number of artists %d" % user_art_mat.ArtistID.count())
artists_tags


# ## Part 1: Finding genres by clustering
# 
# The first task we will look at is how to discover artist genres by only looking at data from plays on Last.fm. One of the ways to do this is to use clustering. To evaluate how well our clustering algorithm performs we will use the user-generated tags and compare those to our clustering results. 
# 
# ### 1.1 Data pre-processing
# 
# Last.fm allows users to associate tags with every artist (See the [top tags](http://www.last.fm/charts/toptags) for a live example). However as there are a number of tags associated with every artists, in the first step we will pre-process the data and get the most popular tag for an artist.

# #### Exercise 1
# 
# **a**. For every artist in **artists_tags** calculate the most frequently used tag. 

# In[55]:

# TODO Implement this. You can change the function arguments if necessary
# Return a data structure that contains (artist id, artist name, top tag) for every artist
#columns = ['DRIVER LIC', 'TAX', 'AMOUNT','TYPE', 'TOLLS']
#ReWeek1tb = pd.DataFrame(Week1tb, columns=columns)
import numpy as np
def calculate_top_tag(all_tags):
    Max_count = all_tags.groupby('ArtistID').apply(
        lambda func: func[func.Count==func.Count.max()]).reset_index(drop=True)
    cols = ['ArtistID', 'ArtistName', 'Tag']
    Ref_artist_tags = pd.DataFrame(Max_count, columns = cols)
    return (Ref_artist_tags)



top_tags = calculate_top_tag(artists_tags)

# Print the top tag for Nirvana
# Artist ID for Nirvana is 5b11f4ce-a62d-471e-81fc-a69a8278c7da
# Should be 'Grunge'
print ("Top tag for Nirvana is %s" %  
       top_tags[top_tags['ArtistID'] == "5b11f4ce-a62d-471e-81fc-a69a8278c7da"].Tag.item())




# In[56]:

user_art_mat.head()


# **b**. To do clustering we will be using `numpy` matrices. Create a matrix from **user_art_mat** with every row in the matrix representing a single artist. The matrix will have 846 columns, one for whether each user listened to the artist.

# In[57]:


def create_user_matrix(input_data):
    return (np.matrix(np.matrix(input_data)[:,1:]))


user_np_matrix = create_user_matrix(user_art_mat)
# Should be (17119, 846)
print (user_np_matrix.shape )

user_np_matrix


# ### 1.2 K-Means clustering
# 
# Having pre-processed the data we can now perform clustering on the dataset. In this assignment we will be using the python library 
# [scikit-learn](http://scikit-learn.org/stable/index.html) for our machine learning algorithms. scikit-learn provides an extensive
# library of machine learning algorithms that can be used for analysis. Here is a [nice flow chart](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) that shows various algorithms implemented
# and when to use any of them. In this part of the assignment we will look at K-Means clustering
# 
# > **Note on terminology**: "samples" and "features" are two words you will come across frequently when you look at machine learning papers or documentation. "samples" refer to data points that are used as inputs to the machine learning algorithm. For example in our dataset each artist is a "sample". "features" refers to some representation we have for every sample. For example the list of 1s and 0s we have for each artist are "features". Similarly the bag-of-words approach from the previous homework produced "features" for each document.
# 
# #### K-Means algorithm
# 
# Clustering is the process of automatically grouping data points that are similar to each other. In the [K-Means algorithm](http://en.wikipedia.org/wiki/K-means_clustering) we start with `K` initially chosen cluster centers (or centroids). We then compute the distance of every point from the centroids and assign each point to the centroid. Next we update the centroids by averaging all the points in the cluster. Finally, we repeat the algorithm until the cluster centers are stable.
# 
# ### Running K-Means
# 
# #### K-Means interface
# Take a minute to look at the scikit-learn interface for calling [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). The constructor of the KMeans class returns a `estimator` on which you can call [fit](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit) to perform clustering.
# 
# #### K-Means parameters
# From the above description we can see that there are a few parameters which control the K-Means algorithm. We will look at one parameter specifically, the number of clusters used in the algorithm. The number of clusters needs to be chosen based on domain knowledge of the data. As we do not know how many genres exist we will try different values and compare the results.
# 
# #### Timing your code
# We will also measure the performance of clustering algorithms in this section. You can time the code in a cell using the **%%time** [IPython magic](http://nbviewer.ipython.org/github/ipython/ipython/blob/1.x/examples/notebooks/Cell%20Magics.ipynb) as the first line in the cell. 
# 
# >**Note**: By default, the scikit-learn KMeans implementation runs the algorithm 10 times with different center initializations. For this assignment you can run it just once by passing the `n_init` argument as 1.

# #### Exercise 2
# 
# **a**. Run K-means using *5* cluster centers on the `user_np_matrix`.

# In[58]:

get_ipython().run_cell_magic('time', '', 'from sklearn.cluster import KMeans\n\n# Run K-means using 5 cluster centers on user_np_matrix\nkmeans_5 = KMeans(n_clusters = 5, n_init =1)\nkmeans_5.fit(user_np_matrix)')


# **b**. Run K-means using *25* and *50* cluster centers on the `user_np_matrix`. Also measure the time taken for both cases.

# In[59]:

get_ipython().run_cell_magic('time', '', 'kmeans_25 = KMeans(n_clusters = 25, n_init =1)\nkmeans_25.fit(user_np_matrix)')


# In[60]:

get_ipython().run_cell_magic('time', '', 'kmeans_50 = KMeans(n_clusters = 50, n_init =1)\nkmeans_50.fit(user_np_matrix)')


# **c**. Of the three algorithms, which setting took the longest to run ? Why do you think this is the case ?

# kmeans_50 algorithm took the longest to run and this is becuase its computing 50 clusters and it has to average  the observations 5X the kmeans_5 algorithm averaged.

# ### 1.3 Evaluating K-Means
# 
# In addition to the speed comparisons we also wish to compare how good our clusters are. To do this we are first going to look at internal evaluation metrics. For internal evaluation we only use the input data and the clusters created and try to measure the quality of clusters. We will use a standard metric for this:
# 
# #### Inertia
# Inertia is a metric that is used to estimate how close the data points in a cluster are. This is calculated as the sum of squared distance for each point to it's closest centroid, i.e., its assigned cluster center. The intution behind inertia is that clusters with lower inertia are better as it means closely related points form a cluster.Inertia is calculated by scikit-learn by default.
# 

# **Exercise 3**

# **a**. Print inertia for all the kmeans model computed above.

# In[61]:

print ("Inertia for KMeans with 5 clusters = %lf " % kmeans_5.inertia_)
print ("Inertia for KMeans with 25 clusters =  %lf " % kmeans_25.inertia_)
print ("Inertia for KMeans with 50 clusters = %lf " % kmeans_50.inertia_)


# **b**. Does KMeans run with 25 clusters have lower or greater inertia than the ones with 5 or 50 clusters ? Why do you think this is ?

# Kmeans_25 clusters has a lower inertia than Kmeans_5 clusters and it also has a slightly higher inertia than Kmeans_50 and this partyl because with Kmeans_25 & Kmeans_50, there are lots of observations (points) which are likely closer to each other; hence minimizing a criterion (inertia). 

# ### Silhouette Plot: 
# A silhouette plot shows a lot of information about the quality of a clustering. You can read more about it <a href="http://en.wikipedia.org/wiki/Silhouette_%28clustering%29">here</a>.
# 
# Each sample x is assigned a score based on two distances:
# * a is the mean distance from x to other elements in its own cluster. 
# * b is the mean distance from x to other elements in the next closest cluster to x.
# then the silhouette score is defined as $$\frac{b-a}{\max(a,b)}$$
# 
# We want this number to be large, since it means the distances within the cluster are smaller than to the next best cluster. 
# To construct the silhouette plot, we first group all the samples by the cluster containining them. 
# Then within each group, we sort all the samples in that cluster by descending silhouette score.
# 
# We start with the labels of the classified points:

# In[62]:

labels = kmeans_5.labels_
len(labels)


# Then we compute the silhouette score for each sample. You dont need to do this for every sample, so define a variable n which is the number of samples to work with from now on. Use the sklearn "silhouette_samples" function to get the silhouette score for those n samples, and create a variable with the corresponding n labels:

# In[63]:

from sklearn.metrics import silhouette_samples

n=1000

slabels = labels[:n]
silhouette_samples_k5  = silhouette_samples(user_np_matrix[:n],slabels)
silhouette_samples_k25 = silhouette_samples(user_np_matrix[:n],kmeans_25.labels_[:n])
silhouette_samples_k50 = silhouette_samples(user_np_matrix[:n],kmeans_50.labels_[:n])

sscore_5  = np.mean(silhouette_samples_k5)
sscore_25 = np.mean(silhouette_samples_k25)
sscore_50 = np.mean(silhouette_samples_k50)

print("Silhouette score for 5 clusters is: %lf" %sscore_5 )
print("Silhouette score for 25 clusters is: %lf" %sscore_25)
print("Silhouette score for 50 clusters is: %lf" %sscore_50)

len(silhouette_samples_k25)


# The next step is to organize the scores into groups by the label. Create a DataFrame with two columns "ClusterId" and "Silhouette Score", and fill it with the n labels and n scores from above. Sort the frame first by ClusterId, then by Silhouette Score, so that cluster samples are grouped together. Now extract the sorted Silhouette Score column as a matrix. Plot it using pylab's "barh" routine, which plots horizontal bars. You should also print out a dataframe with counts of samples for each cluster.

# In[98]:

from pandas import DataFrame
from pylab import *

cols = ['ClusterId', 'Silhouette Score']
df = pd.DataFrame(columns = cols) 
df['ClusterId'] = slabels[:]
df['Silhouette Score'] = silhouette_samples_k5[:]


df_sort = df.sort_values(['ClusterId','Silhouette Score'])
sorted_sscore = df_sort["Silhouette Score"].as_matrix()
plt.barh(np.arange(len(sorted_sscore)), sorted_sscore)
df.head()


# **c** As the clusters gets big, the Silhouette Score gets better.

# >TODO: Answer the question

# ### 1.4 External Evaluation
# While internal evaluation is useful, a better method for measuring clustering quality is to do external evaluation. This might not be possible always as we may not have ground truth data available. In our application we will use `top_tags` from before as our ground truth data for external evaluation. We will first compute purity and accuracy and finally we will predict tags for our **test** dataset.
# 
# #### Exercise 4
# 
# **a**. As a first step we will need to **join** the `artist_tags` data with the set of labels generated by K-Means model. That is, for every artist we will now have the top tag, cluster id and artist name in a data structure.

# In[65]:

# Return a data structure that contains artist_id, artist_name, top tag, cluster_label for every artist
def join_tags_labels(artists_data, user_data, kmeans_model):
    user_data['ClusterID'] = pd.Series(kmeans_model.labels_)
    user_data = user_data[['ArtistID', 'ClusterID']]
    return pd.merge(artists_data, user_data, on='ArtistID')
    #pass
#top_tags.head()
# Run the function for all the models
kmeans_5_joined  = join_tags_labels(top_tags, user_art_mat, kmeans_5 )
kmeans_25_joined = join_tags_labels(top_tags, user_art_mat, kmeans_25)
kmeans_50_joined = join_tags_labels(top_tags, user_art_mat, kmeans_50)

#top_tags.head()
#user_art_mat.head()
#len(kmeans_model.labels_)
#top_tags.head()


# **b**. Next we need to generate a genre for every cluster id we have (the cluster ids are from 0 to N-1). You can do this by **grouping** the data from the previous exercise on cluster id. 
# 
# One thing you might notice is that we typically get a bunch of different tags associated with every cluster. How do we pick one genre or tag from this ? To cover various tags that are part of the cluster, we will pick the **top 5** tags in each cluster and save the list of top-5 tags as the genre for the cluster.
# 

# In[66]:

# Return a data structure that contains cluster_id, list of top 5 tags for every cluster
def assign_cluster_tags(joined_data):
    
    top_cluster = joined_data.groupby(['ClusterID', 'Tag'])#.apply(lambda grp: grp['Count'].count())
    top_cluster = top_cluster.apply(lambda x: x.count())
    top         = top_cluster[['ArtistID']]
    top.columns = ['Count']
    top         = top.sort(columns='Count', ascending=False)
    top         = top.reset_index()
    top         = top.groupby('ClusterID').apply(lambda x: x.head(5))
    top         = top.reset_index(drop=True)
    top         = top[['ClusterID', 'Tag']]
    return top
    
kmeans_5_genres  = assign_cluster_tags(kmeans_5_joined)
kmeans_25_genres = assign_cluster_tags(kmeans_25_joined)
kmeans_50_genres = assign_cluster_tags(kmeans_50_joined)


# #### Cluster Purity
# **Purity** measures the frequency of data belonging to the same cluster sharing the same class label i.e. if we have a number of items in a cluster how many of those items have the same label ? 

# **c**. Compute the purity for each of our K-Means models. To do this find the top tags of all artists that belong to a cluster. Check what fraction of these tags are covered by the top 5 tags of the cluster. Average this value across all clusters. **HINT**: We used similar ideas to get the top 5 tags in a cluster. 

# In[67]:

def get_cluster_purity(joined_data):
    ratio = {}
    top = assign_cluster_tags(joined_data)
    for _, r in joined_data.iterrows():
        cluster = r.ClusterID
        tag = r.Tag
        count = len(top[(top.ClusterID == cluster) & (top.Tag == tag)])
        if cluster in ratio:
            ratio[cluster] = list(map(sum, zip(ratio[cluster], [count, 1])))
        else:
            ratio[cluster] = [count, 1]
    avg = 0.0
    for _, v in ratio.items():
        avg += float(v[0])/v[1]
    purity = avg / len(ratio)
    return purity
    
print ("Purity for KMeans with 5 centers %lf "  % get_cluster_purity(kmeans_5_joined) )
print ("Purity for KMeans with 25 centers %lf " % get_cluster_purity(kmeans_25_joined))
print ("Purity for KMeans with 50 centers %lf " % get_cluster_purity(kmeans_50_joined))


# **d.** What do the numbers tell you about the models? Do you have a favorite?

# The clusters purity are less ** 0.2 ** apart eventhough the second cluster has 5 times more centers than the first cluster; therefore, if we wish to achieve the most purified data we would have to choose 50 centers which would give us the most data points with same group.

# ### 1.5 Evaluating Test Data
# Finally we can treat the clustering model as a multi-class classifier and make predictions on external test data. To do this we load the test data file **userart-mat-test.csv** and for every artist in the file we use the K-Means model to predict a cluster. We mark our prediction as successful if the artist's top tag belongs to one of the five tags for the cluster. 
# 
# #### Exercise 5

# **a** Load the testdata file and create a NumPy matrix named user_np_matrix_test.

# In[68]:

user_art_mat_test = parse_user_artists_matrix(DATA_PATH + "/userart-mat-test.csv")
# NOTE: the astype(float) converts integer to floats here
user_np_matrix_test = create_user_matrix(user_art_mat_test).astype(float)

user_np_matrix_test.shape # Should be (1902, 846)


# **b.** For each artist in the test set, call **[predict](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.predict)** to get the predicted cluster. Join the predicted labels with test artist ids. Return 'artist_id', 'predicted_label' for every artist in the test dataset.

# In[69]:

# For every artist return a list of labels
def predict_cluster(test_data, test_np_matrix, kmeans_model):
    predicted = pd.Series(kmeans_model.predict(test_np_matrix))
    artists   = test_data.ArtistID
    retval    = pd.concat([artists, predicted], axis=1)
    retval.rename(columns={0: 'ClusterID' }, inplace=True)
    return retval
    #pass

# Call the function for every model from before
kmeans_5_predicted  = predict_cluster(user_art_mat_test, user_np_matrix_test, kmeans_5)
kmeans_25_predicted = predict_cluster(user_art_mat_test, user_np_matrix_test, kmeans_25)
kmeans_50_predicted = predict_cluster(user_art_mat_test, user_np_matrix_test, kmeans_50)
#predict_cluster(user_art_mat_test, user_np_matrix_test, kmeans_5)


# **c**. Get the tags for the predicted genre and the tag for the artist from `top_tags`. Output the percentage of artists for whom the top tag is one of the five that describe its cluster. This is the *recall* of our model.
# >NOTE: Since the tag data is not from the same source as user plays, there are artists in the test set for whom we do not have top tags. You should exclude these artists while making predictions and while computing the recall.

# In[70]:

# Calculate recall for our predictions
def verify_predictions(predicted_artist_labels, cluster_genres, top_tag_data):
    #pass
    data = pd.merge(predicted_artist_labels, top_tag_data)
    col_d = ['ArtistID', 'ClusterID', 'Tag']
    data  = pd.DataFrame(data, columns = col_d)
    clusters = data.ClusterID
    tags = data.Tag
    correct = 0.0
    total = len(clusters)
    
    for idx in range(total):
        top_5 = cluster_genres[cluster_genres.ClusterID == clusters[idx]]['Tag']
        
        for tag in top_5:
            if tag == tags[idx]:
                correct += 1.0
                break
    recall = correct / total
    return recall


# **d**. Print the recall for each KMeans model. We define recall as num_correct_predictions / num_artists_in_test_data

# In[71]:

# Use verify_predictions for every model
print ("Recall of KMeans with 5 centers  %lf " %verify_predictions(kmeans_5_predicted, kmeans_5_genres, top_tags))
print ("Recall of KMeans with 25 centers %lf " %verify_predictions(kmeans_25_predicted, kmeans_25_genres, top_tags))
print ("Recall of KMeans with 50 centers %lf " %verify_predictions(kmeans_50_predicted, kmeans_50_genres, top_tags))

#verify_predictions(kmeans_5_predicted, kmeans_5_genres, top_tags)


# # Part 2 - Regression Models - Predicting Song Popularity
# 
# In this section of the assignment you'll be building a model to predict the number of plays a song will get. Again, we're going to be using scikit-learn to train and evaluate regression models, and pandas to pre-process the data.
# 
# In the process, you'll encounter some modeling challenges and we'll look at how to deal with them.
# 
# We've started with the same data as above, but this time we've pre-computed a number of song statistics for you.
# 
# These are:
# 
# 1. plays - the number of times a song has been played.
# 1. pctmale - percentage of the plays that came from users who self-identified as "male".
# 1. age - average age of the listener.
# 1. country1 - the country of the users that listened to this song most.
# 1. country2 - the country of the users that listened to this song second most.
# 1. country3 - the country of the users that listened to this song third most.
# 1. pctgt1 - Percentage of plays that come from a user who's played the song more than once.
# 1. pctgt2 - Percentage of plays that come from a user who's played the song more than twice.
# 1. pctgt5 - Percentage of plays that come from a user who's played the song more than five times.
# 1. cluster - The "cluster number" of the artist associated with this song - similar to what you came up with above. We chose 25 clusters fairly arbitrarily.
# 
# ### 2.1 Data Exploration
# #### Exercise 6
# 
# **a**. Let's start by loading up the data - we've provided a "training set" and a "validation set" for you to test your models on. The training set are the examples that we use to create our models, while the validation set is a dataset we "hold out" from the model fitting process, we use these examples to test whether our models accurately predict new data.
# 

# In[72]:

get_ipython().magic('pylab inline')
import pandas as pd

train = pd.read_csv(DATA_PATH + "/train_model_data.csv")
validation = pd.read_csv(DATA_PATH + "/validation_model_data.csv")


# Now that you've got the data loaded, play around with it, generate some descriptive statistics, and get a feel for what's in the data set. For the categorical variables try pandas ".count_values()" on them to get a sense of the most likely distributions (countries, etc.). 
# 
# **b**. In the next cell put some commands you ran to get a feel for the data.

# In[73]:

train.head()
validation.head()
validation.shape
train.shape


# **c**. Next, create a pairwise scatter plot of the columns: plays, pctmale, age, pctgt1, pctgt2, pctgt5. (_Hint: we did this in lab!_)
# 
# Do you notice anything about the data in this view? What about the relationship between plays and other columns?

# In[74]:

df = train.ix[:,:'pctgt5']
pd.scatter_matrix(df, alpha=0.2, figsize=(14,7) )


# In ***pctgt5*** the corresponding view for **pctgt1 and pctgt2** are inverted but all others remained the same. Similary, the pctgt2 of corresponding view pctgt1 is inverted as well. For all others, the distribution is basically the same.

# ### 2.2 Data Prep and Intro to Linear Regression
# 
# *scikit-learn* does a number of things very well, but one of the things it doesn't handle easily is categorical or missing data. Categorical data is data that can take on a finite set of values, e.g. a categorical variable might be the color of a stop light (Red, Yellow, Green), this is in contrast with continuous variables like real numbers in the range -Infinity to +Infinity. There is another common type of data called "ordinal" that can be thought of as categorical data that has a natural ordering, like: Cold, Warm, Hot. We won't be dealing with this kind of data here, but having that kind of ranking opens up the use of certain other statistical methods.
# 
# 
# #### Exercise 7
# 
# **a**. For the first part of the exercise, let's eliminate categorical variables, and *impute* missing values with pandas. Write a function to drop all categorical variables from the data set, and return two pandas data frames:
# 
# 1. A data frame with all categorical items and a user-specified response column removed.
# 2. A data frame that contains only the response column.

# In[75]:

def basic_prep(data, col):
    #TODO - make a copy of the original dataset but with the categorical variables removed! *Cluster* should be thought of as a 
    #categorical variable and should be removed! Make use of pandas ".drop" function.
    data = data.drop(['country1', 'country2', 'country3', 'cluster'], axis=1)
    #TODO - impute missing values with the mean of those columns, use pandas ".fillna" function to accomplish this.
    means = {'plays':data.plays.mean(), 'pctmale':data.pctmale.mean(), 'age':data.age.mean(), 'pctgt1':data.pctgt1.mean(),
               'pctgt2':data.pctgt2.mean(), 'pctgt5':data.pctgt5.mean(), 'account_age':data.account_age.mean()}
    data = data.fillna(value=means)
    return  data.select(lambda x: x != col, axis=1), data.ix[:,col]

#This will create two new data frames, one that contains training data - in this case all the numeric columns,
#and one that contains response data - in this case, the "plays" column.
train_basic_features, train_basic_response = basic_prep(train, 'plays')
validation_basic_features, validation_basic_response = basic_prep(validation, 'plays')


# Now, we're going to train a linear regression model. This is likely the most widely used model for fitting data out there today - you've probably seen it before, maybe even used it in Excel. The goal of linear modeling, is to fit a **linear equation** that maps a set of **input features** to a numerical **response**. This equation is called a **model**, and can be used to make predictions about the response of similar input features. For example, imagine we have a dataset of electricity prices ($p$) and outdoor temperature ($t$), and we want to predict, given temperature, what electricity price will be. A simple way to model this is with an equation that looks something like $p = basePrice + factor*t$. When we **fit** a model, we are estimating the parameters ($basePrice$ and $factor$) that best fit our data. This is a very simple linear model, but you can easily imagine extending this to situations where you need to estimate several parameters.
# 
# >**Note**: It is possible to fill a semester with linear models (and classes in other departments do!), and there are innumerable issues to be aware of when you fit linear models, so this is just the tip of the iceberg - don't dismiss linear models outright based on your experiences here!
# 
# A linear model models the data as a **linear combination** of the model and its weights. Typically, the model is written with something like the following form: $y = X\theta + \epsilon$, and when we fit the model, we are trying to find the value of $\theta$ that minimizes the **loss** of the model. In the case of regression models, the loss is often represented as $\sum (y - X\theta)^2$ - or the squared distance between the prediction and the actual value.
# 
# In the code below, `X` refers to the the training features, `y` refers to the training response, `Xv` refers to the validation features and yv refers to the validation response. Note that `X` is a matrix (or a `DataFrame`) with the shape $n \times d$ where $n$ is the number of examples and $d$ is the number of features in each example, while `y` is a vector of length $n$ (one response per example).
# 
# Our goal with this assignment is to accurately estimate the number of plays a song will get based on the features we know about it.
# 
# The score we'll be judging the models on is called $R^2$, which is a measure of how well the model fits the data. It can be thought of roughly as the percentage of the variance that the model explains. 
# 
# #### Exercise 9
# 
# **a.** Fit a `LinearRegression` model with scikit-learn and return the model score on both the training data and the validation data.

# In[76]:

from sklearn import linear_model

def fit_model(X, y):
    #TODO - Write a function that fits a linear model to a dataset given a column of values to predict.
    return linear_model.LinearRegression().fit(np.asarray(X)[:,2:], y)

def score_model(model, X, y, Xv, yv):
    #TODO - Write a function that returns scores of a model given its training 
    #features and response and validation features and response. 
    #The output should be a tuple of two model scores.
    return (model.score(np.asarray(X)[:,2:], y), model.score(np.asarray(Xv)[:,2:], yv))

def fit_model_and_score(data, response, validation, val_response):
    #TODO - Given a training dataset, a validation dataset, and the name of a column to predict, 
    #Using the model's ".score()" method, return the model score on the training data *and* the validation data
    #as a tuple of two doubles.
    return score_model(fit_model(data, response), data, response, validation, val_response)
    #END TODO

print (fit_model_and_score(train_basic_features, train_basic_response, validation_basic_features, validation_basic_response))

model = fit_model(train_basic_features, train_basic_response)


# We realize that this may be your first experience with linear models - but that's a pretty low $R^2$ - we're looking for scores significantly higher than 0, and the maximum is a 1. 
# 
# So what happened? Well, we've modeled a **linear** response to our input features, but the variable we're modeling (plays) clearly has a non-linear relationship with respect to the input features. It roughly follows a **power-law** distribution, and so modeling it in linear space yields a model with estimates that are way off.
# 
# We can verify this by looking at a plot of the model's residuals - that is, the difference between the training responses and the predictions. A good model would have residuals with two properties:
# 
# 1. Small in absolute value. 
# 1. Evenly distributed about the true values.
# 
# **b.** Write a function to calculate the residuals of the model, and plot those with a histogram.
# 

# In[77]:

def residuals(features, y, model):
    #TODO - Write a function that calculates model residuals given input features, ground truth, and the model.
    return y-model.predict(np.asarray(features)[:,2:])

#TODO - Plot the histogram of the residuals of your current model.
residuals(train_basic_features, train_basic_response, model).hist(bins=50, figsize = (14, 7))


# See the structure in the plot? This means we've got more modeling to do before we can call it a day! It satisfies neither of our properties - we're often way wrong with our predictions, and seem to systematically **under** predict the number of plays a song will get.
# 
# What happens if we try and predict the $log$ of number of plays? This controls the exponential behaviour of plays, and gives less weight to the case where our prediction was off by 100 when the true answer was 1000. 
# 
# #### Exercise 8
# **a.** Adapt your model fitting from above to fit the **log** of the nubmer of plays as your response variable. Print the scores.

# In[78]:

from sklearn import linear_model

#TODO - Using what you built above, build a model using the log of the number of plays as the response variable.
train_log_response = log(train_basic_response)
valid_log_response = log(validation_basic_response)
print (fit_model_and_score(train_basic_features, train_log_response, validation_basic_features, valid_log_response))
model = fit_model(train_basic_features, train_log_response)
print (model.coef_)


# **b.** You should see a significantly better $R^2$ and validation $R^2$, though still pretty low. Take a look at the model residuals again, do they look any better?

# In[79]:

#TODO Plot residuals of your log model. Note - we want to see these on a "plays" scale, not a "log(plays)" scale!
residuals(train_basic_features, train_log_response, model).hist(bins=50, figsize = (14,7))


# There must be something we can do here to build a better model. Let's try incorporating country and cluster information.
# 
# ### 2.3 Linear Modeling with Categorical Variables: One-Hot Encoding
# 
# Linear models expect **numbers** for input features. But we have some features that we think could be useful that are **discrete** or **categorical**. How do we represent these as numbers?
# 
# One solution is something called one-hot encoding. Basically, we map a discrete space to a vector of binary indicators, then use these indicators as numbers.
# 
# For example, if I had an input column that could take on the values {$RED$, $GREEN$, $BLUE$}, and I wanted to model this with one-hot-encoding, I could use a map:
# 
# * $RED = 001$
# * $GREEN = 010$
# * $BLUE = 100$
# 
# We use this representation instead of traditional binary numbers to keep these features independent of one another.
# 
# Once we've established this representation, we replace the columns in our dataset with their one-hot-encoded values. Then, we can fit a linear model on the data once it's encoded this way!
# 
# Statisticians and econometricians call these types of binary variables *dummy variables*, but we're going to call it one-hot encoding, because that sounds cooler.
# 
# Scikit-learn has functionality to transform values to this encoding built in, so we'll leverage that. The functionality is called `DictVectorizer` in scikit-learn. The idea is that you feed a `DictVectorizer` a bunch of examples of your data (that's the `vec.fit` line), and it builds a map from a categorical variable to a one-hot encoded vector like we have in the color example above. Then, you can use this object to translate from categorical values to sequences of numeric ones as we do with `vec.transform`. In the example below, we fit a vectorizer on the training data and use the same vectorizer on the validation data so that the mapping is consistent and we don't run into issues if the categories don't match perfectly between the two data sets.
# 
# #### Exercise 9
# **a.** Use the code below to generate new training and validation datasets for the datasets *with* the categorical features in them.

# In[80]:

from sklearn import feature_extraction

def one_hot_dataframe(data, cols, vec=None):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a tuple comprising the data, and the fitted vectorizor.
        
        Based on https://gist.github.com/kljensen/5452382
    """
    if vec is None:
        vec = feature_extraction.DictVectorizer()
        vec.fit(data[cols].to_dict(outtype='records'))
    
    vecData = pd.DataFrame(vec.transform(data[cols].to_dict(outtype='records')).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    
    data = data.drop(cols, axis=1)
    data = data.join(vecData)
    return (data, vec)

def prep_dset(data, col, vec=None):
    #Convert the clusters to strings.
    new_data = data
    new_data['cluster'] = new_data['cluster'].apply(str)
    
    #Encode the data with OneHot Encoding.
    new_data, vec = one_hot_dataframe(new_data, ['country1','cluster'], vec)

    #Eliminate features we don't want to use in the model.
    badcols = ['country2','country3','artid','key','age']
    new_data = new_data.drop(badcols, axis=1)
    
    new_data = new_data.fillna(new_data.mean())
    
    return (new_data.drop([col], axis=1), pd.DataFrame(new_data[col]), vec)
    


# In[81]:

train_cats_features, train_cats_response, vec = prep_dset(train, 'plays')
validation_cats_features, validation_cats_response, _ = prep_dset(validation, 'plays', vec)


# **b.** Now that you've added the categorical data, let's see how it works with a linear model!

# In[82]:

print (fit_model_and_score(train_cats_features, train_cats_response, validation_cats_features, validation_cats_response))


# You should see a much better $R^2$ for the training data, but a much *worse* one for the validation data. What happened?
# 
# This is a phenomenon called **overfitting** - our model has too many degrees of freedom (one parameter for each of the 100+ features of this dataset. This means that while our model fits the training data reasonably well, but at the expense of being too specific to that data.
# 
# John Von Neumann famously said ["With four parameters I can fit an elephant, and with five I can make him wiggle his trunk!"](http://www.johndcook.com/blog/2011/06/21/how-to-fit-an-elephant/).
# 
# ### 2.4 Non-linear modeling and Regression Trees
# So, we're at an impasse. We didn't have enough features and our model performed poorly, we added too many features and our model looked good on training data, but not so good on test data.
# 
# What's a modeler to do? 
# 
# There are a couple of ways of dealing with this situation - one of them is called **regularization**, which you might try on your own (see `RidgeRegression` or `LassoRegression` in scikit-learn), another is to use a model which captures **non-linear** relationships between the features and the response variable.
# 
# One such type of model was pioneered here at Berkeley, by the late, great Leo Breiman. These models are called **regression trees**.
# 
# The basic idea behind regression trees is to recursively partition the dataset into subsets that are *similar with respect to the response variable*. 
# 
# If we take our temperature example, we might observe a non-linear relationship - electricity gets expensive when it's cold outside because we use the heater, but it also gets expensive when it's too hot outside because we run the air conditioning. 
# 
# A decision tree model might dynamically elect to split the data on the temperature feature, and estimate high prices both for hot and cold, with lower prices for more Berkeley-like temperatures. Go read the [scikit-learn decision trees documentation](http://scikit-learn.org/stable/modules/tree.html) for more background.
# 
# #### Exercise 10
# **a.** Using the scikit learn `DecsionTreeRegressor` API, write a function that fits trees with the parameter 'max_depth' exposed to the user, and set to 10 by default.

# In[83]:

from sklearn import tree

def fit_tree(X, y, depth=10):
    ##TODO: Using the DecisionTreeRegressor, train a model to depth 10.
    return tree.DecisionTreeRegressor(max_depth=depth).fit(np.asarray(X)[:,2:], y)


# **b.** You should be able to use your same scoring function as above to compute your model scores. Write a function that fits a tree model to your training set and returns the model's score for both the training set and the validation set.

# In[84]:

def fit_model_and_score_tree(train_features, train_response, val_features, val_response):
    ##TODO: Fit a tree model and report the score on both the training set and test set.
    return score_model(fit_tree(train_features, train_response), train_features, train_response, val_features, val_response)


# **c.** Report the scores on the training and test data for both the basic features and the categorical features.

# In[85]:

print (fit_model_and_score_tree(train_basic_features, train_basic_response, validation_basic_features, validation_basic_response))
print (fit_model_and_score_tree(train_cats_features, train_cats_response, validation_cats_features, validation_cats_response))


# Hooray - we've got a model that performs well on the training data set *and* the validation dataset. Which one is better? Why do you think that is. Try varying the depth of the decision tree (from, say, 2 to 20) and see how either data set does with respect to training and validation error. 
# 
# **d.** Now, let's build a tree to depth 3 and take a look at it.

# In[95]:

import pydot


# In[96]:

from io import StringIO
import pydotplus
from IPython.display import Image

tmodel = fit_tree(train_basic_features, train_basic_response, 3)

def display_tree(tmodel):
    dot_data = StringIO()
    tree.export_graphviz(tmodel, out_file=dot_data) 
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    return Image(graph.create_png())
    
display_tree(tmodel)


# **e.** What is the tree doing? It looks like it's making a decision on variables X[4] and X[2] - can you briefly describe, in words, what the tree is doing?

# Based on X[2] and [4], the tree is making a decison on what samples size to use. Observing the values of X[2] and X[4], it looks like it ranges from 0-1 and which suggest that it might be a probabilistic value, prediction value or purity value and base on that value, the sample size gets bigger or smaller. So with X[2] greater than or equal to 0.5025, the sample gets biggger anything less than that the size gets smaller or shrinks.

# **f.** Finally, let's take a look at variable importance for a tree trained to 10 levels - this is a more formal way of deciding which features are important to the tree. The metric that scikit-learn calculates for feature importance is called GINI importance, and measures how much total 'impurity' is removed by splits from a given node. Variables that are highly discriminitive (e.g. ones that occur frequently throughout the tree) have higher GINI scores. You can read more about these scores [here](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#giniimp).

# In[97]:

tmodel = fit_tree(train_basic_features, train_basic_response, 10)


pd.DataFrame(tmodel.feature_importances_, train_basic_features.columns[2:])


# **g.** What do you notice? Is the output interpretable? How would you explain this to someone?

# I noticed that ***10%*** of males has an average age of ***3*** years and that also corresponds with the account_age which makes absolutely no sense because what that says is: there exist a male that started listening to a songs when he was about 1 year old; hence, this output is not interpretable because it makes no sense and should be discarded.
