Professional Machine Learning Engineer Exam Objectives
Frame ML problems
Architect ML solutions
Prepare and process data
Develop ML models
Automate & orchestrate ML pipelines
Monitor, optimize, and maintain ML solutions

Share Google Professional Machine Learning Engineer Sample Questions
NO.1 You are an ML engineer at a global shoe store. You manage the ML models for the company's website. You are asked to build a model that will recommend new products to the user based on their purchase behavior and similarity with other users. What should you do?
A. Build a collaborative-based filtering model
B. Build a classification model
C. Build a regression model using the features as predictors
D. Build a knowledge-based filtering model
Answer: A

NO.2 You have been asked to develop an input pipeline for an ML training model that processes images from disparate sources at a low latency. You discover that your input data does not fit in memory. How should you create a dataset following Google-recommended best practices?
A. Convert the images to tf .Tensor Objects, and then run tf. data. Dataset. from_tensors ().
B. Convert the images to tf .Tensor Objects, and then run Dataset. from_tensor_slices{).
C. Convert the images Into TFRecords, store the images in Cloud Storage, and then use the tf. data API to read the images for training
D. Create a tf.data.Dataset.prefetch transformation
Answer: C


NO.3 You work for an online retail company that is creating a visual search engine. You have set up an end-to-end ML pipeline on Google Cloud to classify whether an image contains your company's product. Expecting the release of new products in the near future, you configured a retraining functionality in the pipeline so that new data can be fed into your ML models. You also want to use Al Platform's continuous evaluation service to ensure that the models have high accuracy on your test data set. What should you do?
A. Keep the original test dataset unchanged even if newer products are incorporated into retraining
B. Extend your test dataset with images of the newer products when they are introduced to retraining
C. Replace your test dataset with images of the newer products when they are introduced to retraining.
D. Update your test dataset with images of the newer products when your evaluation metrics drop below a pre-decided threshold.
Answer: C

NO.4 You are developing a Kubeflow pipeline on Google Kubernetes Engine. The first step in the pipeline is to issue a query against BigQuery. You plan to use the results of that query as the input to the next step in your pipeline. You want to achieve this in the easiest way possible. What should you do?
A. Use the BigQuery console to execute your query and then save the query results Into a new BigQuery table.
B. Write a Python script that uses the BigQuery API to execute queries against BigQuery Execute this script as the first step in your Kubeflow pipeline
C. Locate the Kubeflow Pipelines repository on GitHub Find the BigQuery Query Component, copy that component's URL, and use it to load the component into your pipeline. Use the component to execute queries against BigQuery
D. Use the Kubeflow Pipelines domain-specific language to create a custom component that uses the Python BigQuery client library to execute queries
Answer: A

NO.5 You manage a team of data scientists who use a cloud-based backend system to submit training jobs. This system has become very difficult to administer, and you want to use a managed service instead. The data scientists you work with use many different frameworks, including Keras, PyTorch, theano. Scikit-team, and custom libraries. What should you do?
A. Set up Slurm workload manager to receive jobs that can be scheduled to run on your cloud infrastructure.
B. Create a library of VM images on Compute Engine; and publish these images on a centralized repository
C. Configure Kubeflow to run on Google Kubernetes Engine and receive training jobs through TFJob
D. Use the Al Platform custom containers feature to receive training jobs using any framework
Answer: A

def LRUCache(strArr):
  CACHE_SIZE = 5
  CACHE_DELIMITER = '-'

  cache = []
  for element in strArr:
    if element in cache: 
      cache.append(element)
      # remove first occurance of value in list
      cache.remove(element)
    else:
      cache.append(element)

  cacheString = CACHE_DELIMITER.join(cache[-CACHE_SIZE:])
  return cacheString


# keep this function call here 
print(LRUCache(input()))