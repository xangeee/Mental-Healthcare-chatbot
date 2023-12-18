# Mental-Healthcare-chatbot
<p align="center">
<img src="https://user-images.githubusercontent.com/65236981/170895055-3b6017d2-5fdc-41c5-9a99-1e76bb013f33.png" width="550" height="400"/>
 </p>
In today's society mental health is important because it can impact your thoughts, behaviors and emotions. Living a healthy, balanced life can promote productivity and effectiveness in every our daily activity. Mental health problems are common but help is available. The objective of this project is to create a chatbot that can support people's mental health and wellbeing.

## Brief description of the chatbot
It is built using Natural Language Toolkit(NLTK),Term Frequency-Inverse Document Frequency (TF-IDF), TensorFlow  and Cosine similarity.

The dataset used to train the neural network was obtained from Kaggle ([Mental Health](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)), which contains FAQs about Mental Health.

## Personality prediction model
Given a twitter username the chatbot can also do personality prediction from user's last 1000 tweets. In this project, Machine Learning technique XGBoost is used  to predict four personality traits based on Myers-Briggs Type Indicator(MBTI) model: Introversion-Extroversion(I-E), iNtuitionSensing(N-S), Feeling-Thinking(F-T) and Judging-Perceiving(J-P).

The dataset used for the training was also obtained from Kaggle([Myers-Briggs Personality Type Dataset]([https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot](https://www.kaggle.com/datasets/datasnaek/mbti-type))). As the problem of imbalanced classes with respect to different personality traits is present in the dataset, to solve it the technique Over-sampling was applied for better performance.


## References
<a id="1">[1]</a> 
Khan, A. S., Ahmad, H., Asghar, M. Z., Saddozai, F. K., Arif, A., & Khalid, H. A. (2020). Personality classification from online text using machine learning approach. International Journal of Advanced Computer Science and Applications, 11(3), 460-476.


