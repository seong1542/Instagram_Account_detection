# Instagram_Account_detection
These days, more and more people cheat on SNS. More accounts have been made to trick people into asking for money by sending direct DMs, following them, and sending DMs to them. To detect these fraudulent accounts, I tried to determine fake accounts through machine learning, DNN models, and CNN models by collecting images and csv files from Instagram.

# Statistical data
We personally researched real and fake accounts on Instagram and made them into statistics. We investigated 5,000 real accounts and 5,000 fake accounts, and used a total of 10,000 Train datasets.

![train계정](https://user-images.githubusercontent.com/66362713/120820524-98d4a500-c58f-11eb-8030-54e0801fb8c8.PNG)

The photo shows that the ratio is 1:1.

# Properties

We have attributed things that we can check when we look at our Instagram account. Statistical data has a total of 17 attributes (profile, ID length, number of numbers in ID, name length, name length, name and ID are the same, introduction length, URL presence, number of posts, followers, number of highlights, account tags in introduction, hashtags, professional account presence or not).
(Real=0, Fake=1)

![속성들](https://user-images.githubusercontent.com/66362713/120826828-cde3f600-c595-11eb-975f-c3da21f75efc.PNG)


# Use MachineLearning
We have used a total of six classification models: Random Forest, GradintBoosting, Logistic Regression, XGBoost, LGBM, and Decision Tree. We set the learning data and verification data at an 8:2 ratio. We tabulated the accuracy of the learning data and the accuracy of the verification data when six models were rotated.

*

And the properties are originally made up of int-type.

First, We put it in to three models(RandomForest, LGBM, GradientBoosting) and run it when it was consists only of int types.
Secondly, the properties were executed by changing only 0 and 1 parts into categories or bool types.
Thirdly, only the fake part was changed to the bool type.

**As a result, it had the best accuracy when it consisted only of int types.**

*

Based on the accuracy of the verification data, the three models, Random Forest, LGBM, and Gradient Boosting, had good accuracy. **We tried optuna to improve accuracy with these three models, but the accuracy remained unchanged.**

Then, with these three models, we subtract only one attribute and turn it back to the model to see which attribute affects the accuracy, and we see a difference from the original accuracy. **I could see that the number of followers and followers affected the accuracy the most.**

*

# Correlation
We looked at the correlation because we were curious about the relationship between each attribute. The correlation between properties is like this.

![상관관계](https://user-images.githubusercontent.com/66362713/120826767-be64ad00-c595-11eb-9279-a7e3d7d7aaf8.PNG)


# DeepLearning - DNN

I used DNN as a deep learning model using statistical data. If you put the data into the DNN as it is, the accuracy will be about 50%. 
**I think it's overfitting problem, so I scaled it to solve it.**

* I've tried StandardScaler and MinMaxScaler, and there's a ridiculous accuracy in StandardScaler. Using the MinMaxScaler, I got the right accuracy.
Therefore, I adopted and used the MinMaxScaler.
* Sequential was used in the DNN model to stack layers, giving a different number of hidden units on each layer.
* We used 'relu' as a function of batch normalization and activation on each layer.
* The most accurate layer was four out of the (4,5,6) layers.
* Adam's learning rate was best when 0.001 of (0.01, 0.005, 0.001).
* Dropout was used on each floor, with the lowest loss value of 0.3 out of (0.1, 0.2, 0.3, 0.4)
* The batch size was best when it was 64 out of (32, 64, 128).

As a result, four layers were stacked with sequential, using 'relu' as a function of batch normalization and activation between each layer, Adam's learning rate was 0.001, Dropout was 0.3, and batch size was 64, which showed the lowest loss value and the highest accuracy.

![딥러닝](https://user-images.githubusercontent.com/66362713/120828593-aaba4600-c597-11eb-9815-a621005279bc.PNG)


We have compiled the above in a file. The file name is '인스타그램_딥러닝_실험_softmax기준.ipynb'


**However, statistical data show that machine learning is more accurate than DNN.**




