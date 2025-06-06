# Meal Recommender System for People with Diabetes
## Abstract
Diabetes is a significant, ongoing issue that greatly affects individuals’ diets and
meal plans. However, existing meal recommender systems fail to adapt to changing
user blood glucose levels or are limited to food item recommendations. In this
work, a novel meal recommender system for people with diabetes is proposed
which takes into account both user preferences and user health to calculate and rank
meal scores. Additionally, a custom loss function is proposed to regulate post meal
blood glucose levels in accordance with health guidelines. Experimental results
indicate that the proposed system effectively regulates post meal blood glucose
levels, but with a slight decrease in the preference-based ranking accuracy. With
further refinement, this system has the potential to help users with diabetes better
balance healthy eating with personal meal satisfaction.

## Introduction
Over the years, diabetes has emerged as one of the most pressing challenges in public health,
profoundly affecting the lives and daily routines of millions of people worldwide. In 2021, an
estimated 38.4 million people were found to have diabetes in America alone, making it the 8th leading
cause of death in the United States [1]. In order to avoid further health complications, it is essential
that a person with diabetes consistently monitor their food consumption and changing blood glucose
levels. One promising solution in this area that could aid with this issue is the application of machine
learning.

One such emerging application of machine learning developed in recent years is that of recommender
systems. These systems have numerous applications in a variety of different fields, such as e-
commerce, webpage recommendations, or even meal recommendations. By utilizing user interactions,
such as user reviews or clickthrough rate, recommender systems can predict and learn user preferences
so that it can recommend items that the user will most likely enjoy. In meal recommender systems,
such systems learn each user’s tastes through the use of recipe details and user interactions. However,
when it comes to the specific needs of a person with diabetes, many meal recommender systems
either simply do not support such needs or treat diabetes constraints as a simple filter, limiting
recommendations to meals with little to no sugar. Other diabetes-specific recommender systems lack
user preference considerations or are limited to recommending individual food items over complete
meals. Systems also do not adapt to changing user data, such as blood glucose levels [2]. By creating
a recommender system that dynamically adapts to changing user data, taking into consideration both
user preferences and health constraints, users with diabetes will be able to better regulate their blood
glucose levels while also having an easier time adhering to diets that better align with their personal
tastes. In this work, a novel meal recommender system for people with diabetes is proposed that
integrates both user preferences and health considerations into one final meal ranking score while
utilizing a custom loss function to regularize post meal blood glucose frequencies.

## Related Work
The research by Chen et al. provides a detailed summary of the various techniques developed for
rating prediction and ranking [3]. Some techniques, as detailed by the authors, fall into the content-
based filtering category, in which specific item features are extracted and leveraged by recommender
systems to identify and recommend other similar items. Such item features can be derived from
numerous data sources, such as item textual descriptions or even user reviews. Other techniques
use the collaborative filtering approach, in which user behavior is used to make predictions on item
ratings. Collaborative filtering models can be divided into two main categories: memory based and
model based collaborative filtering. Memory based collaborative filtering utilizes historical rating
data to make predictions about future ratings. For example, it may utilize past interactions from
similar or neighboring users to make rating predictions for a particular user. In contrast, model based
collaborative filtering focuses on training a parametric model for rating and ranking prediction. A
common approach in model based collaborative filtering is to apply matrix factorization to user and
item embeddings, and combine this result with additional rating information, such as the global
rating mean, to calculate a final rating prediction. Hybrid approaches aim to combine several of these
techniques into a single system [3].

Previous research has explored recommender system applications for meal recommendations. One
such system developed by Shandilya et al., called MATURE, combines mandatory user health
constraints with content-based filtering techniques [4]. In this recommender system, user profile
information detailing specific health conditions and their associated health guidelines are used to
filter out meals which are not recommended for their given condition. Once these meals have
been filtered out, traditional recommender system techniques are applied on the remaining meals
to rank and recommend them based on user preferences. Although this approach is generalizable
to multiple health conditions, it may sometimes overlook details specific to certain conditions that
change the types of meals that the user needs to consume. For example, for people with diabetes,
the MATURE recommender system filters out meals with medium to high sugar levels [4]. While
this is generally recommended, blood glucose fluctuations frequently experienced by people with
diabetes can sometimes cause hypoglycemia, in which the user needs to consume food and meals
with high sugar content as opposed to low or zero sugar content. Similarly, the systems developed
by Toledo et al., Stefanidis et al, and Baek et al., all use pre-filtering techniques based on general
health recommendations instead of using specific, changing user data [5, 6, 7]. Systems such as these
provide very little flexibility in the types of meals the user is recommended, basing recommendations
strictly in generalizations of the underlying health condition. Systems like the ones proposed by Baek
et al. and Ramesh et al. only recommend individual food items as opposed to complete meals [7, 8].

## Methodology
In this proposed recommender system, individual preference and health scores are calculated for each
meal, and are then combined together in a weighted sum to calculate the final meal score.

### Preference score
To calculate the preference score, traditional recommender system techniques are used. More specifically,
a hybrid recommender system is used, where model based and memory based collaborative
filtering techniques, as described by Chen et al., are used to predict user recipe ratings and obtain a
preference score [3]. The predicted ratings from the model based and memory based collaborative
filtering approaches are then combined together as a weighted average to obtain the final rating
prediction.

#### Embeddings
First, before the rating predictions can be calculated, embeddings for each recipe and each user need
to be calculated. For the recipe embeddings, all data columns in the Food.com RAW_recipes.csv
file, except for the “id” column, are used to create a ‘document’ for that recipe by joining all of the
information together as text [9]. Next, a Doc2Vec model is created and trained with the created recipe
documents so that it can be used to output recipe embeddings.

To create the user embeddings, the weighted average of the embeddings for the user’s rated recipes is
calculated. The weights in this weighted average are determined by the ratings, on a scale of zero to
five, given to each recipe by the user. Recipes which the user has not given a rating for are left out of
the user’s embedding calculation.

#### Model Based Collaborative Filtering
Now that the user and recipe embeddings have been created, the rating predictions can now be
calculated. First, there is the model based collaborative filtering approach, where the predicted rating
$$\hat{r}$$ by user $$u$$ for recipe $$i$$ is defined by the following equation [3]:

$$\hat{r}_{u, i} = \mu + b_i + b_u + q_i^Tp_u$$

In this equation, $$\mu$$ represents the global rating mean, $$b_i$$ and $$b_u$$ represent trainable biases for recipe $$i$$
and user $$u$$ respectively, $$q_i$$ represents the embedding for recipe $$i$$, and $$p_u$$ represents the embedding for
user $$u$$. This method predicts a user’s rating for a recipe by comparing the user’s embedding, which
represents the user’s overall tastes, to the embedding of the recipe, which represents the recipe’s
qualities. The more similar these embeddings are, the higher the predicted rating. The global rating
mean here serves as a baseline for the rating prediction and the biases are used to further refine the
prediction depending on the particular user or recipe.

#### Memory Based Collaborative Filtering
Next, there is the memory based collaborative filtering approach, where the predicted rating $$\hat{r}$$ by user
$$u$$ for recipe $$i$$ is defined by the following equation [3]:

$$\hat{r}_{u, i} = \overline{r}_u + \frac{\sum _{v \in Neighbors(u)}(r _{v,i}-\overline{r} _v)\times sim(u,v)}{\sum _{v \in Neighbors(u)}|sim(u,v)|}$$ 

In this equation, $$\overline r_u$$ and $$\overline r_v$$ represent the average rating for user $$u$$ and neighboring user $$v$$ respectively,
$$r_{v,i}$$ represents the rating by neighboring user $$v$$ for recipe $$i$$, and $$sim(u, v)$$ represents the similarity
between users $$u$$ and $$v$$. For this implementation, the cosine similarity function and the user embeddings
are used to calculate the similarity $$sim(u, v)$$. In this method, the user’s average rating is used
as a baseline for the rating prediction, and the weighted average of neighboring users’ ratings is
added onto the final rating prediction. The weights in this weighted average are the similarity scores
between the user and the neighboring user.

#### Weighted average
Once the rating prediction is obtained through both the model based and memory based collaborative
filtering methods, the weighted average of these two ratings is calculated in the following manner:

$$\alpha_u(\hat{r} _{u,i,model}) + (1 - \alpha_u)(\hat{r} _{u,i,memory})$$

In this equation, $$\alpha_u$$ is a trainable parameter for user $$u$$, $$\hat{r} _{u,i,model}$$ is the predicted rating by user $$u$$ for
recipe $$i$$ using model based collaborative filtering, and $$\hat{r} _{u,i,memory}$$ is the predicted rating by user $$u$$
for recipe $$i$$ using memory based collaborative filtering. This sum calculates the final predicted rating
for the user and recipe.

### Health score
To calculate the health score, a pre-calculated glycemic load for a meal and the user’s current
blood glucose levels are used. The health score adapts depending on the user’s blood glucose level,
recommending meals with lower glycemic loads if the user has hyperglycemia or normal blood
glucose levels, and recommending meals with higher glycemic loads if the user has hypoglycemia.

#### Data preprocessing: glycemic loads and blood glucose levels
First, the glycemic loads for each individual meal are calculated and added onto the Food.com recipe
data in the RAW_recipes.csv file [9]. To calculate these glycemic loads $$GL$$, the following equation
is used [14]:

$$GL = \frac{(GI_1 \times Carb_1) + (GI_2 \times Carb_2) + ...}{100}$$

In this equation, $$GI$$ is the glycemic index for a particular ingredient and $$Carb$$ is the amount of
carbohydrates for that particular ingredient. The ingredients for each recipe are obtained from the
Food.com dataset and the glycemic indices and average carbohydrate amounts for each ingredient
are obtained from the University of Sydney’s GI database [9, 10]. Because the exact amount of each
ingredient is unknown, the carbohydrate amount for the entire meal is used to scale down the average
ingredient carbohydrate amounts to values that add up to this total carbohydrate amount. The meal
carbohydrate amounts are also obtained from the Food.com dataset [9].

Additionally, initial blood glucose levels for each user are needed in order to calculate the health
score, something which is not provided by the user data in the Food.com dataset. Because obtaining
datasets with real user blood glucose levels is difficult due to patient privacy laws, the initial blood
glucose levels used in this implementation are randomly generated. The blood glucose levels are
assumed to be in mg/dL, and the generated values are floats in the range from 10 to 350, covering all
five blood glucose level types which are used for loss calculation later on in this implementation.

#### Score calculation
To calculate an adequate health score based on the user’s blood glucose level, blood glucose categories
outlined by the American Diabetes Association and glycemic load levels outlined by the Oregon
State University are used [15, 16]. A blood glucose level greater than 180 mg/dL is considered
hyperglycemia, a blood glucose level between 70 to 180 mg/dL is considered in-range, and a blood
glucose level less than 70 mg/dL is considered hypoglycemia [15]. For the glycemic load, a value
less than or equal to 10 is considered low, a value between 11 to 19 is considered intermediate, and
a value greater than or equal to 20 is considered high [16]. In score calculation, when a user has
hyperglycemia, the assigned health score value is $$e^{−GL}$$ for meals with a glycemic load less than
or equal to 10 and zero otherwise. This way, meals with lower glycemic loads are given higher
scores and meals with medium to high glycemic loads are assigned health scores of zero. Similarly,
if a user has in-range blood glucose levels, then $$e^{−GL}$$ is used as the health score for meals with a
glycemic load less than or equal to 15, and are assigned zero otherwise. Like with hyperglycemia,
lower glycemic loads are assigned higher health scores, but this time, some intermediate glycemic
load values are also taken into consideration. Finally, for users with hypoglycemia, a health score of
$$1 − e^{−GL}$$ is assigned to meals with a glycemic load greater than or equal to 15, and are scored zero
otherwise. Here, higher glycemic loads are assigned higher health scores, and only medium to high
glycemic loads are assigned positive health scores.

### Final meal score
Once both the preference and health scores are calculated, a weighted sum is calculated in the
following manner to obtain a final score for the meal:

$$(\alpha)P + (1 - \alpha)H$$

In this equation, $$P$$ is the preference score, $$H$$ is the health score, and $$\alpha$$ is a trainable weight parameter.
In order to allow the system to further adjust the preference to health score balance in the final meal
score calculation using blood glucose levels, the weight parameter $$\alpha$$ is further divided into the
following components:

$$\alpha=\alpha_{u,hyper2}I_{hyper2}+\alpha_{u,hyper1}I_{hyper1}+\alpha_{u,normal}I_{normal}+\alpha_{u,hypo1}I_{hypo1}+\alpha_{u,hypo2}I_{hypo2}+b_u$$

The indicator values $$I$$ are either zero or one, and are used to indicate whether the blood glucose level
is level two hyperglycemia, level one hyperglycemia, normal, level one hypoglycemia, or level two
hypoglycemia. The indicator values then activate a corresponding trainable weight parameter $$\alpha$$ for
that blood glucose level. This allows for differing preference to health score balances depending on
the user’s blood glucose level.

## Experiments
### Data
#### Food.com Recipes and Interactions dataset
In order to simulate a meal recommender system for people with diabetes, two main datasets were
used. The first is the Food.com Recipes and Interactions dataset provided by the Kaggle website. This
dataset contains over 180,000 recipes and over 700,000 recipe reviews from the Food.com website,
covering a span of 18 years of uploads and user interactions [9]. For this particular system, this
dataset provides the recipes which will be used for meal recommendations and provides information
on the details of each recipe, such as ingredients, nutrition, descriptions, time of preparation, and
more, which may influence a user’s preference towards that meal and may influence the meal’s
glycemic load. This data is obtained from the dataset’s provided RAW_recipes.csv file. Additionally,
the Food.com dataset provides information on user ratings, including information such as the recipe
which the rating was for and the given rating on a scale of zero to five. Data for user ratings can be
found in the provided interactions_test.csv, interactions_train.csv, and interactions_validation.csv
files. The information provided by these files are used as testing, training, and validation data to train
the system for meal recommendations on a basis of user preferences [9].

#### University of Sydney’s GI database
The second dataset used is the University of Sydney’s GI Database, available on the university’s
glycemic index research and GI news website. This dataset contains the glycemic indices and average
carbohydrate portions for 4,269 different food items [10]. Because this data was only available on the
associated website, the database information was retrieved using the Selenium and Beautiful Soup
python libraries for use in the meal recommender system [11, 12]. The main purpose of this dataset is
to provide carbohydrate and glycemic index information to be used alongside the recipes information
from the Food.com dataset, so that the recipe glycemic loads can be estimated. These glycemic loads
are then used, along with user blood glucose levels, to recommend meals on a basis of user health.
Because datasets detailing user blood glucose levels are not widely available, for the purposes of
testing and training this recommender system, the blood glucose levels for each user are randomly
generated.

#### USDA Food Composition database
Originally, the USDA Food Composition database was also going to be used alongside the previously
mentioned datasets to obtain specific carbohydrate amounts for each recipe ingredient using the
indicated amounts on the Food.com dataset [13]. Using the exact carbohydrate for each recipe
ingredient would have led to more accurate glycemic load calculations. However, it was found
that very few of the listed recipes on the Food.com dataset actually provided the amounts of each
ingredient needed for each recipe. Due to this lack of information, the USDA Food Composition
database is no longer used in the implementation of this meal recommender system. Some functions
for retrieving and manipulating the USDA database data have already been implemented prior to
this decision, so they will still be visible and available in the source code for this project. In the
final implementation, the average carbohydrate amounts provided by the University of Sydney’s GI
Database are used for glycemic load calculations instead.

### Experimental setup
All models used in this experiment trained for a maximum of 25 epochs, used a batch size of 256,
used the Adam optimizer, and used a learning rate of 0.001. Because the data used is better suited
for rating prediction tasks, the Mean Squared Error (MSE) loss function was used on value-based
models, such as Graph Convolutional Matrix Completion (GCMC), Deep Matrix Factorization
(DMF), and Matrix Factorization (MF). Ranking models that utilized the BPR loss function, such as
Bayesian Personalized Ranking (BPR) and Neural Matrix Factorization (NeuMF), were also tested,
but ultimately not used as a baseline comparison due to their reliance on negative samples. All
baselines in this experiment were obtained using the RecBole python library [17, 18, 19]. All models
used the same training, validation, and testing data split, as provided by the Food.com dataset [9].

#### Custom Loss Function
In order the train the proposed health and preference recommender system, a custom loss function
was used combining MSE and post meal blood glucose frequency loss calculations as shown in the
following equation:

$$Loss=MSE+max(0,f_{hyper2}-0.05)+max(0,f_{hyper1}-0.25)+max(0,0.7-f_{normal}) + max(0,f_{hypo1}-0.04)+max(0,f_{hypo2}-0.01)$$

Here, $$f$$ is the frequency for a particular post meal blood glucose level. If a blood glucose frequency
lies above or below the associated recommended percentage, then the difference between the percentages
is assigned as the loss value. These percentages have been taken from the American Diabetes
Association guidelines, which recommend these percentage goals in order to avoid future health
complications. For most adults with type one or type two diabetes, they recommend that they spend
less than 5% of their time in level two hyperglycemia, less than 25% of their time in level one
hyperglycemia, more than 70% of their time within in-range blood glucose levels, less than 4% of
their time in level one hypoglycemia, and less than 1% of their time in level two hypoglycemia [15].
These percentages are used for the loss calculation.

### Experimental results
Figure 1 shows the training results for the health and preference recommender system proposed in
this paper. For the 25 epochs used, the model was successfully able to train and converge on the
used dataset. Similarly, Figure 2 and Figure 3 show the training results for the GCMC and DMF
models respectively, which were also able to converge on the used dataset. The MF model in Figure
4, while it was able to converge on the training data, showed minimal improvement on the validation
data. Originally, BPR was also going to be used as a baseline for comparison, but upon further
inspection of its training results, it was found that the model was not able to properly generalize to
the validation data. The training results for the BPR model are found in Figure 5. Yet, even with this
poor generalization, for the NDCG@10 metric, the model achieved a score of 1.0 on the testing data.
These results are most likely due to the BPR model’s sensitivity to negative samples. Because the
Food.com dataset does not provide negative samples, they are created using unseen items from the
training dataset. However, for BPR to train properly, the negative samples need to represent negative
samples from the entire dataset, giving BPR a slight advantage on unseen data in comparison to the
value-based models. Additionally, the NDCG@10 score of 1.0 is mostly due to the lack of negative
samples in the testing data. Due to these major differences between the BPR model and the other
tested models, BPR will no longer be used as a baseline comparison in this experiment. The NeuMF
model faced similar issues when using the BPR loss function, as seen in Figure 6, and was also not
used as a baseline for comparison.

<div align="center">
  
![Figure_1](https://github.com/user-attachments/assets/24dcd5a0-8729-4556-8aeb-f91811dd59a5) 

Figure 1: Proposed model training.

</div>

<br />

<div align="center">

![Figure_2](https://github.com/user-attachments/assets/063fd9a6-8e95-4081-b5d7-c541582ad901)

Figure 2: GCMC model training.

</div>

<br />

<div align="center">

![Figure_3](https://github.com/user-attachments/assets/e32bca9d-1d6d-4475-91bf-a6b1e2a4d2f8)

Figure 3: DMF model training.

</div>

<br />

<div align="center">

![Figure_4](https://github.com/user-attachments/assets/5a991e4c-9943-4720-a3f4-947a1d1c8219)

Figure 4: MF model training.

</div>

<br />

<div align="center">

![Figure_5](https://github.com/user-attachments/assets/c51cf75a-b811-4c12-9fe8-0e3e4c0fcc36)

Figure 5: BPR model training.

</div>

<br />

<div align="center">

![Figure_6](https://github.com/user-attachments/assets/10846df6-7872-4e3f-8e44-15bca20ff9c1)

Figure 6: NeuMF model training.

</div>

For each model, the NDCG@10 and post meal blood glucose frequencies were calculated and used
as metrics for model comparison. The results for each model can be found in Table 1.

<div align="center">

Table 1: Baseline comparison results

|**Model**|**NDCG@10**|**Hyper2_Freq**|**Hyper1_Freq**|**Normal_Freq**|**Hypo1_Freq**|**Hypo2_Freq**|
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|**Proposed**|0.767|29.1%|24.5%|36.7%|4.2%|5.5%|
|**DMF**|1.0|41%|15.2%|35.8%|4%|4%|
|**MF**|0.833|31.6%|16.8%|40.2%|5.6%|5.8%|
|**GCMC**|0.9|35.2%|14%|41.2%|4.4%|5.2%|

</div>

## Discussion
There are two important limitations to address regarding the testing results. The first one is on the
distribution of the initial user blood glucose levels. As mentioned previously, the initial blood glucose
levels for each user were randomly generated due to a lack of real blood glucose data. Because of
this, the initial blood glucose level frequencies could end up having unrealistic distributions between
the different blood glucose level types, and could even already start at dangerous percentages, such as
with really high hypoglycemia or hyperglycemia frequencies. To obtain a better comparison between
the different model frequencies and better understand how each model is adjusting the final blood
glucose frequencies, during testing, it was ensured that an equal number of randomly generated
blood glucose levels were produced for each blood glucose category. Using this method, the initial
frequencies for each of the five categories were set at 20%.

Another limitation lies in the blood glucose adjustments for hyperglycemia. Using a meal, the user’s
current blood glucose level can either only be maintained, using meals with a glycemic load of zero,
or can be increased, using meals with a higher glycemic load. Therefore, the post meal blood glucose
frequency for level two hyperglycemia will always have a percentage equal to or higher than the
initial frequency, which in this case is 20%, and will not be able to adhere to the recommended
frequency of less than 5% [15]. A similar problem arises with level one hyperglycemia. While level
one hyperglycemia can maintain its post meal frequency below the recommended 25%, a meal cannot
bring hyperglycemia down to an in-range blood glucose value. In total, level one and level two
hyperglycemia will be stuck with a post meal blood glucose frequency of 40% and above, making
it impossible for the in-range post meal blood glucose frequency to have the recommended 70% or
above frequency [15].

Overall, the proposed health and preference recommender system has increased regularization on
post meal blood glucose levels when compared to other baselines, particularly for the hyperglycemia
categories. As seen in Table 1, the proposed system has an increased balance between the level one
and level two hyperglycemia frequencies, with level two hyperglycemia receiving a 9.1% increase
from the base 20%. On the other hand, baselines which have a higher NDCG@10, such as DMF, have
a significantly higher level two hyperglycemia, with an increase of 21%. All four tested models were
able to keep the level one hyperglycemia frequency below the 25% guideline. While the proposed
model may have the highest level one hyperglycemia frequency, this is most likely attributed to the
model preventing these blood glucose levels from becoming level two hyperglycemia. For level
one and level two hypoglycemia, the proposed model was over the recommended frequencies by
0.2% and 4.5% respectively. This model achieved better level one hypoglycemia frequencies than
both the MF and GCMC models, and a better level two hypoglycemia frequency than the MF model.
Because the model is balancing both user health and user preferences, the reduced normal blood
glucose frequency and increased level two hypoglycemia frequency are most likely due to user meal
preferences. However, further investigation is needed to confirm this. As seen by the NDCG@10
values, incorporating this post meal blood glucose frequency regularization results in reduced meal
ranking accuracy based on user preferences.

There is still plenty of room for improvement in order to have this system better apply to real world
scenarios. First, the cold-start problem still needs to be addressed. Traditional cold-start techniques,
such as the ones described by Chen et al., can be applied in a real system [3]. Next, machine
learning techniques can be applied to post meal blood glucose and glycemic load calculations for
more realistic results. These topics have already been explored by Karim et al. and by Lee et al.,
and can potentially be combined with the proposed health and preference model to create a more
complete system [20, 21]. Finally, in order to ensure proper data collection to improve the testing
of this system, the proposed model would need to be tested with real user data in an actual meal
recommender application. This way, realistic blood glucose frequencies can be obtained, and analysis
can be performed on user data over time. Further exploration can be performed on the long term
impact of this proposed recommender system on user health.

## Conclusion
By combining traditional collaborative filtering techniques with diabetes health guidelines, the
proposed system was able to successfully improve post meal blood glucose frequencies, but at a slight
cost of the preference-based ranking accuracy. The proposed health and preference recommender
system not only can help people with diabetes better manage their health needs, but can also help
them better adhere to their diets by recommending meals they will enjoy. Future directions include
further refinement of the health to preference score balance, integration with other machine learning
techniques, and testing of the system in real-world settings. The code and datasets used in this work
are publicly available at: https://github.com/HannaMG/diabetes-meal-recommender-system.git

## References
[1] American Diabetes Association, “About Diabetes: Statistics,” Diabetes.org. [Online]. Available:
https://diabetes.org/about-diabetes/statistics/about-diabetes

[2] R. Yera, A. A. Alzahrani, L. Martínez, and R. M. Rodríguez, “A systematic review on food recommender
systems for diabetic patients,” Int. J. Environ. Res. Public Health, vol. 20, no. 5, p. 4248, 2023, doi:
10.3390/ijerph20054248.

[3] L. Chen, G. Chen, and F. Wang, “Recommender systems based on user reviews: the state of the art,” User
Model User-Adap. Inter., vol. 25, pp. 99–154, 2015, doi: 10.1007/s11257-015-9155-5.

[4] R. Shandilya, S. Sharma, and J. Wong, “MATURE-Food: Food Recommender System for MAndatory
FeaTURE Choices A system for enabling Digital Health,” International Journal of Information Management
Data Insights, vol. 2, no. 2, art. no. 100090, 2022, doi: 10.1016/j.jjimei.2022.100090.

[5] R. Y. Toledo, A. A. Alzahrani, and L. Martínez, "A Food Recommender System Considering Nutri-
tional Information and User Preferences," IEEE Access, vol. 7, pp. 96695-96711, 2019, doi: 10.1109/AC-
CESS.2019.2929413.

[6] K. Stefanidis, D. Tsatsou, D. Konstantinidis, L. Gymnopoulos, P. Daras, S. Wilson-Barnes, K. Hart, V.
Cornelissen, E. Decorte, E. Lalama, A. Pfeiffer, M. Hassapidou, I. Pagkalos, A. Argiriou, K. Rouskas, S.
Hadjidimitriou, V. Charisis, S. B. Dias, J. A. Diniz, G. Telo, H. Silva, A. Bensenousi, and K. Dimitropoulos,
"PROTEIN AI Advisor: A Knowledge-Based Recommendation Framework Using Expert-Validated Meals for
Healthy Diets," Nutrients, vol. 14, no. 20, Article 4435, 2022, doi: 10.3390/nu14204435.

[7] J.-W. Baek, J.-C. Kim, J. Chun, and K. Chung, “Hybrid clustering based health decision-making for improving
dietary habits,” Technology and Health Care, vol. 27, no. 5, pp. 459–472, 2019, doi: 10.3233/THC-191730.

[8] N. Ramesh, S. Dabbiru, A. Arya, and A. Rehman, “A Novel Rule-Based Recommender System For The Indian
Elderly Diabetic Population,” in Proc. 2021 5th International Conference on Informatics and Computational
Sciences (ICICoS), 2021, pp. 41–46, doi: 10.1109/ICICoS53627.2021.9651768.

[9] S. Li, “Food.com Recipes and Interactions,” Kaggle, 2019. [Online]. Available:
https://www.kaggle.com/dsv/783630. doi: 10.34740/KAGGLE/DSV/783630.

[10] University of Sydney, “GI Search,” Glycemic Index Research and GI News. [Online]. Available:
https://glycemicindex.com/gi-search/.

[11] B. Muthukadan, “Selenium with Python,” [Online]. Available: https://selenium-python.readthedocs.io/.

[12] L. Richardson, “Beautiful Soup Documentation,” [Online]. Available:
https://www.crummy.com/software/BeautifulSoup/bs4/doc/.

[13] U.S. Department of Agriculture, “USDA Food Composition Database,” FoodData Central. [Online].
Available: https://fdc.nal.usda.gov/download-datasets.

[14] Better Health Channel, “Carbohydrates and the glycemic index,” Better Health Channel. [Online]. Available:
https://www.betterhealth.vic.gov.au/health/healthyliving/carbohydrates-and-the-glycaemic-index.

[15] American Diabetes Association Professional Practice Committee, “6. Glycemic Goals and Hypoglycemia:
Standards of Care in Diabetes—2025,” Diabetes Care, vol. 48, Suppl. 1, pp. S128–S145, Jan. 1, 2025, doi:
10.2337/dc25-S006.

[16] J. Higdon, V. J. Drake, B. Delage, and S. Liu, "Glycemic Index and Glycemic Load," Linus Pauling
Institute, Oregon State University, Mar. 2016. [Online]. Available: https://lpi.oregonstate.edu/mic/food-
beverages/glycemic-index-glycemic-load.

[17] W. X. Zhao, S. Mu, Y. Hou, Z. Lin, Y. Chen, X. Pan, K. Li, Y. Lu, H. Wang, C. Tian, Y. Min, Z. Feng, X.
Fan, X. Chen, P. Wang, W. Ji, Y. Li, X. Wang, and J.-R. Wen, “RecBole: Towards a unified, comprehensive and
efficient framework for recommendation algorithms,” in Proc. CIKM, 2021, pp. 4653–4664.

[18] W. X. Zhao, Y. Hou, X. Pan, C. Yang, Z. Zhang, Z. Lin, J. Zhang, S. Bian, J. Tang, W. Sun, Y. Chen, L. Xu,
G. Zhang, Z. Tian, C. Tian, S. Mu, X. Fan, X. Chen, and J.-R. Wen, “RecBole 2.0: Towards a more up-to-date
recommendation library,” in Proc. CIKM, 2022, pp. 4722–4726.

[19] L. Xu, Z. Tian, G. Zhang, J. Zhang, L. Wang, B. Zheng, Y. Li, J. Tang, Z. Zhang, Y. Hou, X. Pan, W. X. Zhao,
X. Chen, and J.-R. Wen, “Towards a more user-friendly and easy-to-use benchmark library for recommender
systems,” in Proc. SIGIR, 2023, pp. 2837–2847.

[20] R. A. H. Karim, I. Vassányi, and I. Kósa, “After-meal blood glucose level prediction using an absorption
model for neural network training,” Computers in Biology and Medicine, vol. 125, art. no. 103956, 2020, doi:
10.1016/j.compbiomed.2020.103956.

[21] H. Lee, M. Um, K. Nam, S.-J. Chung, and Y. Park, "Development of a Prediction Model to Esti-
mate the Glycemic Load of Ready-to-Eat Meals," Foods, vol. 10, no. 11, art. no. 2626, 2021, doi:
10.3390/foods10112626.
