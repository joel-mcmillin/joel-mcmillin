## **Decision Tree - Auto Accident (2016-2023)**

This project involved taking 7.7 million rows of US automobile accident data, preparing it and then using a decision tree model to determine which variables could help in predicting accident severity on a scale of 1-4, with level 1 being least severe and level 4 being the most severe. I used resampling due to the unbalanced dataset, with the overwhelming majority of observations being of level 2 severity, in order to avoid overfitting the model. I also used feature selection to find the top 5 features to refine the model before evaluating it for accuracy.

## **Data Source**

The data used comes from: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

## **References**

* Moosavi, S. (2021). US-Accidents: A Countrywide Traffic Accident Dataset. https://smoosavi.org/datasets/us_accidents 

* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, arXiv preprint arXiv:1906.05409 (2019).

* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. “Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.” In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.
