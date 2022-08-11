### Tips 

from https://towardsdatascience.com/how-i-passed-the-gcp-professional-ml-engineer-certification-47104f40bec5

General

Typical Big Data pipeline for streaming data:

Pub/Sub -> Dataflow -> BigQuery or Cloud Storage

Typical Big Data pipeline for batch data:

Pub/Sub -> Cloud Run or Cloud Functions -> Dataflow -> BigQuery or Cloud Storage

* Use the general use APIs by default (Vision, Video Intelligence, Natural Language…). Only use AutoML if you have custom needs (custom labels, etc.)
* To de-identify sensible data, you can redact, tokenize or hash, using BigQuery, Cloud Storage, Datastore, or Data Loss Protection (DLP)
* Difference between TensorBoard and TensorFlow Model Analysis: the former evaluates during training, based on mini-batches, while the latter evalutes after training, can be done in slices of data and is based on the full data
* AI Explanations: with tabular data, you can use Shapely or integrated ingredients for large feature spaces; with images, you can use integrated gradients for pixel-level explanations or XRAI for region-level explanations.
* When to use Kubeflow over TFX? When you need PyTorch, XGBoost or if you want to dockerize every step of the flow
* Keras: use the Sequential API by default. If you have multiple inputs or outputs, layer sharing or a non-linear topology, change to the Functional API, unless you have a RNN. If that is the case, Keras Subclasses instead
* 3 methods for optimizing TensorFlow pipelines: prefetch, interleave and cache


### BigQuery ML

It supports the following types of model: linear regression, binary and multiclass logistic regression, k-means, matrix factorization, time series, boosted trees, deep neural networks, AutoML models and imported TensorFlow models
Use it for quick and easy models, prototyping etc.

### Storage

Choosing storage for analytics:

Structured data: Bigtable for millisecond latency, BigQuery for latency in seconds
Unstructured: use Cloud Storage by default, and Firebase storage for mobile

### Accelerators

Choosing between CPUs, TPUs and GPUs:

Use CPUs for quick prototypes, simple/small models or if you have many C++ custom operations; use GPU if you have some custom C++ operations and/or medium to large models; use TPUs for big matrix computations, no custom TensorFlow operations and/or very large models that train for weeks or months

To improve performance on TPUs: if data pre-processing is a bottleneck, do it offline as a one-time cost; choose the larges batch size that fits in memory; keep the per-core batch size the same

### Neural networks

Common pitfalls in backpropagation and their solutions:

vanishing gradients -> use ReLu
exploding gradients -> use batch normalization
ReLu layers are dying -> lower learning rates

For multiclass classification, if:

labels and probabilities are mutually exclusive, use softmax_cross_entropy_with_logits_v2
labels are mutually exclusive, but not probabilities, use sparse_softmax_cross_entropy_with_logits
labels are not mutually exclusive, use sigmoid_cross_entropy_with_logits

# Learning Stuff 


### Labs 

 Recommending Products Using Cloud SQL and Spark 
https://www.cloudskillsboost.google/course_sessions/554292/labs/102245



```bash
echo "Authorizing Cloud Dataproc to connect with Cloud SQL"
CLUSTER=rentals
CLOUDSQL=rentals
ZONE=us-central1-f
NWORKERS=2
machines="$CLUSTER-m"
for w in `seq 0 $(($NWORKERS - 1))`; do
   machines="$machines $CLUSTER-w-$w"
done
echo "Machines to authorize: $machines in $ZONE ... finding their IP addresses"
ips=""
for machine in $machines; do
    IP_ADDRESS=$(gcloud compute instances describe $machine --zone=$ZONE --format='value(networkInterfaces.accessConfigs[].natIP)' | sed "s/\['//g" | sed "s/'\]//g" )/32
    echo "IP address of $machine is $IP_ADDRESS"
    if [ -z  $ips ]; then
       ips=$IP_ADDRESS
    else
       ips="$ips,$IP_ADDRESS"   
    fi
done
echo "Authorizing [$ips] to access cloudsql=$CLOUDSQL"
gcloud sql instances patch $CLOUDSQL --authorized-networks $ips
```


### Recommending Products Using Cloud SQL and Spark -- Module Test

1. True or False: Cloud SQL is a big data analytics warehouse

Answer: False -- Correct - Cloud SQL is a transaction RDBMS or relational database management system. It is designed for many more WRITES than READS.Whereas BigQuery is a big data analytics warehouse which is optimized for reporting READS.

2. 
Cloud SQL and Cloud Dataproc offer familiar tools (MySQL and Hadoop/Pig/Hive/Spark). What is the value-add provided by Google Cloud Platform? (Select the 2 correct options below )
 

* Google-proprietary extensions and bug fixes to MySQL, Hadoop, and so on

* It’s the same API, but Google implements it better


* Fully-managed versions of the software offer no-ops
Yes. No-ops is the main value-add here.

* Running it on Google infrastructure offers reliability and cost savings
Yes. You pay only for the resources you use. Cloud SQL can be shut down when it’s not being used. Hadoop clusters can be of preemptible nodes, and so on.

3. You are thinking about migrating your Hadoop workloads to the cloud and you have a few workloads that are fault-tolerant (they can handle interruptions of individual VMs gracefully). What are some architecture considerations you should explore in the cloud? Choose all that apply

* You are thinking about migrating your Hadoop workloads to the cloud and you have a few workloads that are fault-tolerant (they can handle interruptions of individual VMs gracefully). What are some architecture considerations you should explore in the cloud? Choose all that apply


* Migrate your storage from on-cluster HDFS to off-cluster Google Cloud Storage (GCS)
Correct!

* Use PVMs or Preemptible Virtual Machines
Correct!

* Consider having multiple Cloud Dataproc instances for each priority workload and then turning them down when not in use
Correct!


4. True or False: If you are migrating your Hadoop workload to the cloud, you must first rewrite all your Spark jobs to be compliant with the cloud.

Answer: False -- Correct - you can run your same Spark job code running on the same Hadoop software but running on cloud hardware with Cloud Dataproc.


5. Complete the following: You should feed your machine learning model your _______ and not your _______. It will learn those for itself!

data, rules 

6. Relational databases are a good choice when you need:

* Fast queries on terabytes of data

* Streaming, high-throughput writes

* Aggregations on unstructured data

* Transactional updates on relatively small datasets -- correct 

7. Google Cloud Storage is a good option for storing data that: (Select the 2 correct options below).

* Will be accessed frequently and updated constantly with new transactions from a front-end and needs to be stored in a relational database

* Is ingested in real-time from sensors and other devices and supports SQL-based queries


* May be required to be read at some later time (i.e. load a CSV file into BigQuery) -- correct 

* May be imported from a bucket into a Hadoop cluster for analysis -- correct 




### Lab -- Creating a Streaming Data Pipeline for a Real-Time Dashboard with Dataflow 


Task 1. Create a Pub/Sub topic and BigQuery dataset
Task 2. Create a Cloud Storage bucket
Task 3. Set up a Dataflow Pipeline
Task 4. Analyze the taxi data using BigQuery
Task 5. Perform aggregations on the stream for reporting
Task 6. Create a real-time dashboard
Task 7. Create a time series dashboard
Task 8. Stop the Dataflow job


biggest thing is creating datflow pipeline from template and creating aggregate in bigquery 

Task 3. Set up a Dataflow Pipeline

Dataflow is a serverless way to carry out data analysis. In this lab, you set up a streaming data pipeline to read sensor data from Pub/Sub, compute the maximum temperature within a time window, and write this out to BigQuery.

    In the Cloud Console, go to Navigation menu > Dataflow.

    In the top menu bar, click CREATE JOB FROM TEMPLATE.

    Enter streaming-taxi-pipeline as the Job name for your Dataflow job.

    Under Dataflow template, select the Pub/Sub Topic to BigQuery template.

    Under Input Pub/Sub topic, enter projects/pubsub-public-data/topics/taxirides-realtime

    Under BigQuery output table, enter <myprojectid>:taxirides.realtime
    
    Under Temporary location, enter gs://<mybucket>/tmp/


And then use this SQL query to make aggregates


```sql 
WITH streaming_data AS (
SELECT
  timestamp,
  TIMESTAMP_TRUNC(timestamp, HOUR, 'UTC') AS hour,
  TIMESTAMP_TRUNC(timestamp, MINUTE, 'UTC') AS minute,
  TIMESTAMP_TRUNC(timestamp, SECOND, 'UTC') AS second,
  ride_id,
  latitude,
  longitude,
  meter_reading,
  ride_status,
  passenger_count
FROM
  taxirides.realtime
WHERE ride_status = 'dropoff'
ORDER BY timestamp DESC
LIMIT 100000
)
# calculate aggregations on stream for reporting:
SELECT
 ROW_NUMBER() OVER() AS dashboard_sort,
 minute,
 COUNT(DISTINCT ride_id) AS total_rides,
 SUM(meter_reading) AS total_revenue,
 SUM(passenger_count) AS total_passengers
FROM streaming_data
GROUP BY minute, timestamp
```


### Perform Foundational Data, ML, and AI Tasks in Google Cloud: Challenge Lab  (Expert) Lab


Create a simple Dataproc job
Create a simple DataFlow job
Create a simple Dataprep job
Perform one of the three Google machine learning backed API tasks

Task 4: AI

Complete one of the tasks below, YOUR_PROJECT must be replaced with your lab project name.

    Use Google Cloud Speech API to analyze the audio file gs://cloud-training/gsp323/task4.flac. Once you have analyzed the file you can upload the resulting analysis to gs://YOUR_PROJECT-marking/task4-gcs.result.

    Use the Cloud Natural Language API to analyze the sentence from text about Odin. The text you need to analyze is "Old Norse texts portray Odin as one-eyed and long-bearded, frequently wielding a spear named Gungnir and wearing a cloak and a broad hat." Once you have analyzed the text you can upload the resulting analysis to gs://YOUR_PROJECT-marking/task4-cnl.result.

    Use Google Video Intelligence and detect all text on the video gs://spls/gsp154/video/train.mp4. Once you have completed the processing of the video, pipe the output into a file and upload to gs://YOUR_PROJECT-marking/task4-gvi.result. Ensure the progress of the operation is complete and the service account you're uploading the output with has the Storage Object Admin role.



### Invoking ML APIs from AI Platform Notebooks (jupyter notebook) labs 

https://www.cloudskillsboost.google/course_sessions/570479/labs/102982

REAlly cool to see basic usage of some crazy powerful APIs! 

Also noticed there is a new book out (put in amazon cart) for learning about AI on GCP.


### cloud natural language 


score of the sentiment ranges between -1.0 (negative) and 1.0 (positive) and corresponds to the overall emotional leaning of the text.

magnitude indicates the overall strength of emotion (both positive and negative) within the given text, between 0.0 and +inf. Unlike score, magnitude is not normalized; each expression of emotion within the text (both positive and negative) contributes to the text's magnitude (so longer text blocks may have greater magnitudes). 

### LAB  Analyzing data using AI Platform Notebooks and BigQuery

In this lab, you analyze a large (70 million rows, 8 GB) airline dataset using BigQuery and AI Platform Notebooks.

Looking at flights and presenter points out how powerful it is to be able to make aggregates in bigquery and then analyze them later in notebooks. 

for example we have 70M (8GB) records in big query that we then create an aggregate of and can actually plot these in our little Jupyter notebook for much cheaper. 

### LAB Improving Data Quality 

Machine learning models can only consume numeric data, and that numeric data should be 1s or 0s. Data is said to be messy or untidy if it is missing attribute values, contains noise or outliers, has duplicates, wrong data, or upper/lower case column names, or is essentially not ready for ingestion by a machine learning algorithm.

In this lab, you will present and solve some of the most common issues of untidy data. Note that different problems will require different methods, and they are beyond the scope of this notebook.

What you learn

In this lab, you will:

    Resolve missing values.

    Convert the Date feature column to a datetime format.

    Rename a feature column, remove a value from a feature column.

    Create one-hot encoding features.

    Understand temporal feature conversions.


In the notebook interface, navigate to **training-data-analyst > courses > machine_learning > deepdive2 > launching_into_ml > labs, and open improve_data_quality.ipynb** 


Solutions notebook 

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/solutions/improve_data_quality.ipynb



#### Data Quality Issue #5:
Temporal Feature Columns


Our dataset now contains year, month, and day feature columns. Let's convert the month and day feature columns to meaningful representations as a way to get us thinking about changing temporal features -- as they are sometimes overlooked.

Note that the Feature Engineering course in this Specialization will provide more depth on methods to handle year, month, day, and hour feature columns.


```python
# Here we map each temporal variable onto a circle such that the lowest value for that variable appears right next to the largest value. We compute the x- and y- component of that point using the sin and cos trigonometric functions.
df['day_sin'] = np.sin(df.day*(2.*np.pi/31))
df['day_cos'] = np.cos(df.day*(2.*np.pi/31))
df['month_sin'] = np.sin((df.month-1)*(2.*np.pi/12))
df['month_cos'] = np.cos((df.month-1)*(2.*np.pi/12))

# Let's drop month, and day
# TODO 5
df = df.drop(['month','day','year'], axis=1)
```


###  Exploratory Data Analysis Using Python and BigQuery (LAB)

In the notebook interface, navigate to training-data-analyst > courses > machine_learning > deepdive2 > launching_into_ml > labs and open python.BQ_explore_data.ipynb.


### Improve Data Quality - Quiz

1. Which of the following refers to the Orderliness of data?


The data record with specific details appears only once in the database
The data represents reality within a reasonable period
None of the above
x - The data entered has the required format and structure

2. Which of the following are categories of data quality tools?

Cleaning tools 
Monitoring tools
Both A and B 
None of the Above

3. What are the features of low data quality?
Unreliable info
Duplicated data
Incomplete data 
All of the above

4. Which of the following are best practices for data quality management?
Resolving missing values
Automating data entry
Preventing duplicates 
All of the above


5. Which of the following is not a Data Quality attribute?
Consistency
Auditability
Accuracy
x - redundancy 


### Exploratory Data Analysis Using Python and BigQuery - LAB 

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/solutions/python.BQ_explore_data.ipynb

### Quiz: Exploratory Data Analysis 

1. Which of the following is not true about Exploratory Data Analysis?


Discovers new knowledge.
Generates a posteriori hypothesis.
Does not provide insight into the data. - x
Deals with unknowns. 


2. Exploratory Data Analysis is majorly performed using the following methods: 
Bivariate
Univariate
both A & B -x
None of the above

3. What are the objectives of exploratory data analysis?

Gain maximum insight into the data set and its underlying structure.
Check for missing data and other mistakes.
Uncover a parsimonious model, one which explains the data with a minimum number of predictor variables.
All of the above - x



4. Which of the following is not a component of Exploratory Data Analysis?

Anomaly Detection
Accounting and Summarizing
Statistical Analysis and Clustering
Hyperparameter tuning - x 

5. Which is the correct sequence of steps in data analysis and data visualisation of Exploratory Data Analysis?  

Data Exploration -> Data Cleaning -> Model Building -> Present Results - x
Data Exploration -> Data Cleaning -> Present Results -> Model Building
Data Exploration -> Model Building -> Present Results -> Data Cleaning
Data Exploration -> Model Building -> Data Cleaning -> Present Results

### Quiz: Supervised Learning 

1. Which model would you use if your problem required a discrete number of values or classes?

Regression Model 
Classification Model - x
Supervised Model 
Unsupervised Model 


2. Which of the following machine learning models have labels, or in other words, the correct answers to whatever it is that we want to learn to predict? 

Unsupervised Model
None of the above.
Reinforcement Model
Supervised Model - x

3. Which statement is true?

Depending on the problem you are trying to solve, the data you have, explainability, etc. will not determine which machine learning methods you use to find a solution.
None of the above
Determining which machine learning methods you use to find a solution depends only on the problem or hypothesis.
Depending on the problem you are trying to solve, the data you have, explainability, etc. will determine which machine learning methods you use to find a solution. - x

4. What is a type of Supervised machine learning model?

Regression model 
Classification model 
Both A & B - x
None of the above

5. When the data isn’t labelled, what is an alternative way of predicting the output? 

Clustering Algorithms -x 
Logistic Regression 
Linear Regression 
None of the above 




###  Introduction to Linear Regression 

training-data-analyst > courses > machine_learning > deepdive2 > launching_into_ml > Labs and open intro_linear_regression.ipynb.

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/solutions/intro_linear_regression.ipynb

###  Quiz: Neural Networks 

1. Which activation functions are needed to get the complex chain functions that allow neural networks to learn data distributions.

Nonlinear activation functions - x
Linear activation functions 
All of the above 
none of the above

2. A single unit for a non-input neuron has ____________________ a/an

Output of the activation function
Activation function
Weighted Sum
all of the above - x

3. Which of the following activation functions are used for nonlinearity?

Tanh
Hyperbolic tangent
Sigmoid
All of the above - x


4. Which activation function has a range between zero and Infinity?

ReLU - x
Tanh
Sigmoid
ELU

5. If we wanted our outputs to be in the form of probabilities, which activation function should I choose in the final layer?

ReLU
Tanh
Sigmoid - x  
ELU 

### Decision trees and Random Forests LAB 

https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/solutions/decision_trees_and_random_Forests_in_Python.ipynb

###  Quiz: Decision Trees AND Random Forests 

1. In a decision classification tree, what does each decision or node consist of? 

Euclidean distance minimizer
Mean squared error minimizer
Linear classifier of one feature - x
 Linear classifier of all features 

2. Which of the following statements is true?

Mean squared error minimizer and euclidean distance minimizer are used in classification, not regression. 
Mean squared error minimizer and euclidean distance minimizer are used in regression, not classification. - x
Mean squared error minimizer and euclidean distance minimizer are not used in regression and classification.
Mean squared error minimizer and euclidean distance minimizer are used in regression and classification.

3. Decision trees are one of the most intuitive machine learning algorithms. They can be used for which of the following?

Regression
Classification
Both A & B -x 
None of the above 


4. A random forest is usually more complex than an individual decision tree; this makes it harder to visually interpret ? 

True - x
False 


### Optimization Quiz 

1. For the formula used to model the relationship i.e. y = mx + b, what does ‘m’ stand for? 


It refers to a bias term which can be used for regression.
It captures the amount of change we've observed in our label in response to a small change in our feature. - x
Both a & b
None of the above

2. What are the basic steps in an ML workflow (or process)?

Check for anomalies, missing data and clean the data
Perform statistical analysis and initial visualization
Collect data 
All of the above - x

3. Which of the following statements is true?

To calculate the Prediction y for any Input value x we have three unknowns, the m = slope(Gradient), b = y-intercept(also called bias) and z = third degree polynomial.
To calculate the Prediction y for any Input value x we have two unknowns, the m = slope(Gradient) and b = y-intercept(also called bias).  - x
None of the above 
To calculate the Prediction y for any Input value x we have three unknowns, the m = slope(Gradient), b = y-intercept(also called bias) and z = hyperplane.

### Optimization Quiz 2 

1. Fill in the blanks: Simply speaking, __________ is the workhorse of basic loss functions. ______ is the sum of squared distances between our target variable and predicted values. 


Log loss
Likelihood
Mean Squared Error - x 
None of the above 


2. Which of the following loss functions is used for classification problems?

MSE 
cross entropy - x
Both A & B 
None of the above 

3. Fill in the blanks: At its core, a ________ is a method of evaluating how well your algorithm models your dataset. If your predictions are totally off, your _________ will output a higher number. If they’re pretty good, it will output a lower number. As you change pieces of your algorithm to try and improve your model, your ______ will tell you if you’re getting anywhere. 

Loss function - x
Bias term
Activation functions
Linear model

4. Loss functions can be broadly categorized into 2 types: Classification and Regression Loss. _____ is typically used for regression and ______ is typically used for classification.

Log Loss, Focus Loss
Mean Squared Error, Cross Entropy - x
Cross Entropy, Log Loss
None of the above

### Optimization Quiz - Gradients 

1. Which of the following gradient descent methods is used to compute the entire dataset? 

Mini-batch gradient descent 
Gradient descent
None of the above 
Batch gradient descent  -x


2. Fill in the blanks. ________________: Parameters are updated after computing the gradient of error with respect to the entire training set ________________: Parameters are updated after computing the gradient of error with respect to a single training example ________________: Parameters are updated after computing the gradient of error with respect to a subset of the training set

Mini Batch Gradient Descent, Batch Gradient Descent, Stochastic Gradient Descent
Mini-Batch Gradient Descent, Stochastic Gradient Descent, Batch Gradient Descent
Batch Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent - x
None of the above

3. Select which statement is true.

Batch gradient descent, also called vanilla gradient descent, calculates the error for each example within the training dataset, but only after all training examples have been evaluated does the model get updated. This whole process is like a cycle and it's called a training epoch. - x

Batch gradient descent, also called vanilla gradient descent, calculates the gain for each example within the training dataset, but only before all training examples have been evaluated does the model get updated. This whole process is like a cycle and it's called a training epoch.

Batch gradient descent, also called vanilla gradient descent, calculates the error for each example within the training dataset, but only before all training examples have been evaluated does the model get updated.

None of the above 

4. Select the correct statement(s) regarding gradient descent.

In machine learning, we use gradient descent to determine if our model labels needs to be de-optimized.

Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.

Gradient descent is an optimization algorithm used to maximize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.

 All of the above 



