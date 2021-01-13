## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset is all about bank marketing data with about 32950 rows containing the job type, marital status, house ownership, in loan debt and many other details employee details. The best performing model was an Auto ML model which was More Efficient Model Training.

## Scikit-learn Pipeline
The data was obtained from a url in a CSV format to an Azure TabularDataset object. The data is then changed over to a Pandas DataFrame cleaned, and the results split into a confined Pandas Series. 

Next, the data is then seprated into test and train sets, with a degree of 20% for the test, and the 80% for train. The Logistic Rregression was utilized from the Scikit-learn pipeline, and the hyperparameters tuned were the Regularization Strength and the Maximum Number of Iterations. 

**Benefits of Random sampler you chosen**
I chose Random sampling because it supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. Some users do an initial search with random sampling and then refine the search space to improve results.

The Parameter Sampler chosen is the Random Sampler that returns sporadic characteristics on a described pursuit space. This procedure is faster, but does not give a better result un-like the Grid Sampler. The Regularization Strength was planned with uniform inspecting, which gives a value reliably dispersed between the base and most extraordinary likely characteristics. It's the most major and safe limit looking at system for incessant components. The Maximum Number of Iterations will be a choice among values some place in the scope of 1 and 1000, with no genuine discrete hyperparameter spread. We should not view this variable as constant. 

The stopping approach chosen was the BanditPolicy, which is a way to exit the process subject to a breathing space factor and evaluation length, it ends when the exactness of a run isn't inside the slack total diverged from the best performing run. It's a less moderate course of action that will be sufficient for our examinations.

When preparing an AI model the principle objective is to get the best performing model that has the best presentation on the approval set. We center around the approval set as it speaks to how well the model sums up (execution on concealed information). Hyperparameters structure the reason of the preparation cycle. For eg., on the off chance that the learning rate is set too high, at that point the model may never meet to the minima as it will make too enormous strides after each emphasis. Then again, if the learning rate is set too low it will require some investment for the model to come to the minima. Hyperparameter tuning can be considered as a discovery advancement issue where we attempt to locate at least a capacity f(x) without knowing its investigative structure. It is likewise called subsidiary free enhancement as we don't have a clue about its expository structure and no subsidiaries can be registered to limit f(x), and henceforth methods like slope drop can't be utilized. 

Strategic relapse is one of the most well-known and valuable characterization calculations in AI which I utilized in the venture. Logistic regression is one of the most common and useful classification algorithms in machine learning which I used in the project. 

Customarily, pipelines include for the time being cluster handling, for example gathering information, sending it through a venture message transport and preparing it to give pre-determined outcomes and direction for following day's activities. While this works in certain enterprises, it is truly inadequate in others, and particularly with regards to ML applications. 

The accompanying graph shows a ML pipeline applied to a constant business issue where highlights and expectations are time delicate (for example Netflix's proposal motors, Uber's appearance time assessment, LinkedIn's associations recommendations, Airbnb's web crawlers and so on)

Parameter Sampling/Boundary inspecting is utilized in an AI gathering calculation called bootstrap collecting (additionally called stowing). It helps in keeping away from overfitting and improves the strength of AI calculations. 

In random sampling, hyperparameter values are randomly selected from the defined search space.

In packing, a specific number of similarly estimated subsets of a dataset are removed with substitution. At that point, an AI calculation is applied to every one of these subsets and the yields are ensembled.

In AI, early halting is a type of regularization used to stay away from overfitting when preparing a student with an iterative technique, for example, inclination drop. Such techniques update the student to improve it fit the preparation information with every emphasis. 

During preparing, the model is assessed on a holdout approval dataset after every age. In the event that the exhibition of the model on the approval dataset begins to debase (for example misfortune starts to increment or precision starts to diminish), at that point the preparation cycle is halted. The model at the time that preparation is halted is then utilized and is known to have great speculation execution. In the event that regularization techniques like weight rot that update the misfortune capacity to support less mind boggling models are considered "express" regularization, at that point early halting might be considered as a sort of "certain" regularization, much like utilizing a more modest organization that has less limit.

## AutoML
Each AI framework has hyperparameters, and the most fundamental errand in mechanized AI (AutoML) is to consequently set these hyperparameters to advance execution. Particularly ongoing profound neural organizations urgently rely upon a wide scope of hyperparameter decisions about the neural organization's design, regularization, and improvement. *AutoML Pipeline* was a *VotingEnsemble* calculation, which is an ensamble model made by *AutoML*. That implies *AutoML* made it's own calculation made out of different calculations: *XGBoostClassifier*, *LightGBM*, and *SGD*, with various weights.The JSON acquired from the run interface completely clarifies the model and it's loads.Computerized hyperparameter advancement (HPO) has a few significant use cases; it can lessen the human exertion essential for applying AI. This is especially significant with regards to AutoML.




## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The *AutoML* model introduced best outcomes, however the model utilizing hyperparameter tuning was not a terrible decision. The large favorable position of utilizing *AutoML* rather than *HyperDrive* was that *AutoML* is quicker to begin with, and right now presents the outcome in a compress document with the conda climate and models prepared to convey. All things considered, boundary tuning is quicker to prepare, and much more adaptable in what you can be tuned.

In an "ordinary" AI model, human intervention and fitness are required at different stages including data ingestion, data pre-getting ready, and desire models. Of course, using AutoML, every movement other than data variety and desire can be automated to make a changed AutoML pipeline for any business customer. Next, what is the prerequisite for making a revamped modernized AI pipeline? The creating interest for AI models by undertakings is driving the improvement of simple to utilize ML structures that can be used ready to move by any business customer. Through its robotization, a changed AutoML pipeline can give the going with focal points: Improve effectiveness of data experts by means of automating any repetitive ML-related endeavors and help them with focusing in on various issues. Reduce human slip-ups in ML models that arise in a general sense due to manual advances. Make AI accessible for all customers, subsequently propelling a decentralized cycle. In the accompanying zone, we will discuss a bit of the well known structures in AutoML (or AutoML mechanical assemblies) that are enabling a creation AI pipeline. Joined on top of the scikit-learn AI group in Python, Auto-Sklearn is the motorized version that frees any ML customer from endeavors, for instance, picking the right ML figuring and tuning the hyperparameters. Fitting for little to medium datasets, Auto-Sklearn can make and improve an AI pipeline using Bayesian journeys.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
In the AutoML model preparation, neural networks were left out which could introduce better results. Neural networks set aside more effort to prepare, but they are more costly. Since we are just testing the efficiency of Auto ML, it would not be optimal to utilize Deep Learning. Future work to improve the pipeline could consider these further developed calculations that could introduce better outcomes, the means would be to empower complex models like profound learning at the AutoML design and also To utilize a profound learning system, (for example, Keras) rather than SKLearn on train.py

