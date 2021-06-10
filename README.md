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

*![머신러닝 정확도](https://user-images.githubusercontent.com/66362713/120845787-3a69ef80-c5ac-11eb-9a79-4d96411a3d89.PNG)



And the properties are originally made up of int-type.

First, We put it in to three models(RandomForest, LGBM, GradientBoosting) and run it when it was consists only of int types.
Secondly, the properties were executed by changing only 0 and 1 parts into categories or bool types.
Thirdly, only the fake part was changed to the bool type.

**As a result, it had the best accuracy when it consisted only of int types.**

*![머신러닝 정확도 표로 정리](https://user-images.githubusercontent.com/66362713/120845752-3047f100-c5ac-11eb-8452-6229bcc17839.PNG)


Based on the accuracy of the verification data, the three models, Random Forest, LGBM, and Gradient Boosting, had good accuracy. **We tried optuna to improve accuracy with these three models, but the accuracy remained unchanged.**

Then, with these three models, we subtract only one attribute and turn it back to the model to see which attribute affects the accuracy, and we see a difference from the original accuracy. **I could see that the number of followers and followers affected the accuracy the most.**

*![영향을 많이 받는 속성들](https://user-images.githubusercontent.com/66362713/120845772-33db7800-c5ac-11eb-8d61-b61a0ba69266.PNG)


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
* The most accurate layer was 5 out of the (4,5,6) layers.
* Adam's learning rate was best when 0.001 of (0.01, 0.005, 0.001).
* Dropout was used on each floor, with the lowest loss value of 0.2 out of (0.1, 0.2, 0.3)
* The batch size was best when it was 64 out of (32, 64, 128).

![dnn최적](https://user-images.githubusercontent.com/66362713/120860205-3c3dae00-c5c0-11eb-9dba-533e97d71d64.PNG)


As a result, 5 layers were stacked with sequential, using 'relu' as a function of batch normalization and activation between each layer, Adam's learning rate was 0.001, Dropout was 0.2, and batch size was 64, which showed the lowest loss value and the highest accuracy.

![딥러닝](https://user-images.githubusercontent.com/66362713/120860050-0f899680-c5c0-11eb-8383-e656ca88e78b.PNG)



We have compiled the above in a file. The file name is '인스타그램_딥러닝_실험_softmax기준.ipynb'


**However, statistical data show that machine learning is more accurate than DNN.**


# DeepLearning - CNN
We used the CNN model using the captured pictures of Instagram accounts as image data. With all the status bars removed from the photos, learning about real and fake accounts showed a 40 to 50% chance.
We used transform to improve this accuracy, Resize, RandomResolvedCrop, RandomHorizontalFlip, RandomAffine, Rotation, ToTensor, Normalize...I tried using etc.
We tried three algorithms: "resnet18", "mobilenet", and "shufflenet".
After trying with a large number of cases, we found that the best accuracy was achieved when using the 'resnet' algorithm, using Resize, RandomResolvedCrop, RandomHorizontalFlip, ToTensor, Normalize.

![cnn결과](https://user-images.githubusercontent.com/66362713/121469278-2cc2c880-c9f7-11eb-98c3-3a7cebfe733f.png)

We tested it by putting 20 real accounts and 20 fake accounts ourselves, and we found that we predicted 32 accounts out of 40 accounts.
