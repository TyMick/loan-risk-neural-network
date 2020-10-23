# Building a Neural Network to Predict Loan Risk

When I heard about LendingClub's [public loan dataset](https://www.kaggle.com/wordsforthewise/lending-club), containing all loans issued by the company from it's launch in 2007 until the end of 2018, I figured that'd be the perfect opportunity to build a predictive model.

I wrote an article detailing my entire process, which you can read on [my blog](https://tymick.me/blog/loan-risk-neural-network "Building a Neural Network to Predict Loan Risk – Ty Mick"), [Towards Data Science](https://towardsdatascience.com/loan-risk-neural-network-30c8f65f052e "Building a Neural Network to Predict Loan Risk | Towards Data Science"), [Hacker Noon](https://hackernoon.com/loan-risk-prediction-using-neural-network-algorithm-gg4q3uu2 "Loan Risk Prediction Using Neural Networks | Hacker Noon"), or [DEV](https://dev.to/tywmick/building-a-neural-network-to-predict-loan-risk-3k1f "Building a Neural Network to Predict Loan Risk - DEV Community"). If you'd like to follow along in your own Jupyter Notebook, you can go ahead and fork mine [on Kaggle](https://www.kaggle.com/tywmick/building-a-neural-network-to-predict-loan-risk) or [here on GitHub](https://github.com/tywmick/loan-risk-neural-network/blob/master/models/loan-risk-neural-network.ipynb).

After building the model itself, I built an API to serve its predictions, using [Flask](https://flask.palletsprojects.com/en/1.1.x/), [TensorFlow](https://www.tensorflow.org/)/[Keras](https://keras.io/), [pandas](https://pandas.pydata.org/), and [scikit-learn](https://scikit-learn.org/). You can interact with the API by either visiting its [demonstrational front end](https://tywmick.pythonanywhere.com/) or sending a GET request directly to `https://tywmick.pythonanywhere.com/api/predict`. The front end site includes a form where you can fill in all the parameters for the API request, and there are a couple of buttons at the top that let you fill the form with typical examples from the dataset (since there are a _lot_ of fields to fill in).

I later wrote a couple of follow-up posts expanding the project:

- [Can I Grade Loans Better Than LendingClub?](https://tymick.me/blog/loan-grading-showdown)
- [Natural Language Processing for Loan Risk](https://tymick.me/blog/loan-risk-nlp)

Please enjoy, and [let me know](https://tymick.me/connect) if you have any questions!
