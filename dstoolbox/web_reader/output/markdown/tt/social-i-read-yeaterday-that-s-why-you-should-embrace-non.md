---
url: https://www.linkedin.com/posts/adrianolszewski_i-read-yeaterday-thats-why-you-should-share-7461451556740218881-B3cR/?utm_source=share&utm_medium=member_android&rcm=ACoAACmphIUBo2D4cUQR_ZXSmdR7KMycDh8BUk8
scraped_at: 2026-07-27T14:09:45.434067
depth: 0
---

I read yeaterday: "That’s why you should embrace non-parametric statistics and start loving ranks 🙂". This is in spirit "Non-parametric = no problems" claim, which I hardly disagree with if made… | Adrian Olszewski



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# Adrian Olszewski’s Post

[Adrian Olszewski](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_feed-actor-name)

2mo

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_i-read-yeaterday-thats-why-you-should-activity-7461451559902830592-ap6w&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

I read yeaterday: "That’s why you should embrace non-parametric statistics and start loving ranks 🙂". This is in spirit "Non-parametric = no problems" claim, which I hardly disagree with if made generally.
Think about it.
What hypothesis will we assess? You lose simple interpretation in most cases.
Let's take the Mann-Whitney(-Wilcoxon): it's not about means, not about medians, not even entire distributions. It's about stochastic superiority and this is not something we easily use, say, in clinical non-inferiority or bioequivalence studies.
Ranks have no connection to any common measure unless a strong (and often unrealistic assumption) holds: identically distributed, leading to shift-location model.
And even this isn't fully true for Mann-Whitney(-Wilcoxon), because if variances are different (which happens almost always), the type-1 error is compromised.
So yes, IT DOES make some assumption 😉
And we need then the Brunner-Munzel just like Welch-Satterthwaite for the t-test case.
But why not employing a distribution-free test that preserve a meaningful outcome, e.g. expressed in means (assuming they DO make sense for such data) like permutation Welch t test, Freedman-Lane AN(C)OVA, wild bootstrap AN(C)OVA? Or quantile regression maybe, giving you well-interpretable outcomes?
This is important for example in my field - clinical trials, where the key analyses must be planned in advance and use concrete estimators aligned with the clinical question. If comparison of means for clinical superiority testing was planned, switching to a rank-based test changes that. In other words, you obtain a technically valid answer to a never asked question.
/ \*it' s easy to find cases where p < 0.0..01 for equal means or medians at low N, and p>0.999 for different means or medians - even at large N. /
Maybe you'll be very lucky and the IID holds + also the distributions are close to Gaussian - but then hey! Probably either version of the CLT would do the job fine, without even reaching for ranks and lowering power or compromising Type-1 error (so important in clinical trials).
Rank-based methods like DO have their deserved place in case of difficult distributions (where a single-number summary doesn't suffice) or ordinal data, but at the cost of harder interpretation.
For ordinal data stochastic superiority (probabilist index) is the most useful information we can obtain, but for numerical data? We can do much better!
Visit my thread on the ResearchGate for a discussion, examples and important literature (books, papers): [https://lnkd.in/dzcR2uAx](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FdzcR2uAx&urlhash=iOVk&trk=public_post-text)
Take my challenge about embracing the permutation Welch t test: [https://lnkd.in/dBwCQz3P](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FdBwCQz3P&urlhash=HaKS&trk=public_post-text)
(Yep, an AI-generated pic, but - surprisingly - quite well handled :) )




[46](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_i-read-yeaterday-thats-why-you-should-activity-7461451559902830592-ap6w&trk=public_post_social-actions-reactions)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_i-read-yeaterday-thats-why-you-should-activity-7461451559902830592-ap6w&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_i-read-yeaterday-thats-why-you-should-activity-7461451559902830592-ap6w&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_i-read-yeaterday-thats-why-you-should-activity-7461451559902830592-ap6w&trk=public_post_feed-cta-banner-cta)

## More Relevant Posts

* [Janani Gopi](https://in.linkedin.com/in/janani0302?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjanani0302_statistics-bayesianinference-researchmethods-activity-7469373364961267712-cM7u&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Two researchers. Same data. Same hypothesis.
  One reports p = 0.03
  The other reports a 94% posterior probability.
  At first glance, it looks like they disagree.
  In reality, they're answering two different questions.
  The Frequentist vs. Bayesian debate isn't just a philosophical exercise. It shapes how we design studies, interpret evidence, and communicate uncertainty.
  A Frequentist asks:
  If there were actually no effect, how surprising would data this extreme be?
  The answer is summarized by a p-value. It measures the compatibility of the observed data with the null hypothesis. It does not tell us the probability that the hypothesis is true.
  A Bayesian asks:
  Given what I knew before and what I've observed now, how should I update my belief?
  The result is a posterior distribution, which can be used to estimate the probability of different hypotheses or effect sizes after seeing the data.
  Why does this distinction matter?
  Clinical trials: Bayesian adaptive designs can incorporate accumulating evidence as data arrives, potentially allowing earlier decisions when evidence becomes compelling.
  A/B testing: A p-value tells us how unusual the observed difference would be under the null hypothesis. A Bayesian analysis can directly estimate the probability that one variant outperforms another.
  Small-sample research: When data are limited, incorporating sensible prior knowledge can help stabilize estimates and reduce uncertainty.
  Does that mean Bayesian methods are better?
  Not necessarily.
  Frequentist methods have a long history, strong theoretical foundations, and remain the standard in many scientific disciplines. Bayesian methods offer a flexible framework for updating evidence and incorporating prior information.
  Neither approach is universally superior.
  The real skill is understanding which question your analysis is actually answering.
  Because the biggest mistake in statistics isn't choosing Frequentist or Bayesian methods.
  It's interpreting the answer from one framework as if it came from the other.
  [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#BayesianInference](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbayesianinference&trk=public_post-text) [#ResearchMethods](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fresearchmethods&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#ScientificMethod](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fscientificmethod&trk=public_post-text)




  [59](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjanani0302_statistics-bayesianinference-researchmethods-activity-7469373364961267712-cM7u&trk=public_post_social-actions-reactions)







  [24 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjanani0302_statistics-bayesianinference-researchmethods-activity-7469373364961267712-cM7u&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjanani0302_statistics-bayesianinference-researchmethods-activity-7469373364961267712-cM7u&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjanani0302_statistics-bayesianinference-researchmethods-activity-7469373364961267712-cM7u&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjanani0302_statistics-bayesianinference-researchmethods-activity-7469373364961267712-cM7u&trk=public_post_feed-cta-banner-cta)
* [Yash Mane](https://in.linkedin.com/in/yashvilasmane?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashvilasmane_machinelearning-datascience-heartdisease-activity-7465792261243428865-5tty&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Hey, I'm Yash Mane.
  This is my series: Learning Machine Learning from Scratch.
  Today's topic: Heart Disease Prediction Foundation Project 2
  Project 1 = Regression (predict a number).
  Project 2 = Classification (predict 0 or 1).
  Same pipeline. Different problem. Let's break it down.
  EDA (Exploratory Data Analysis)
  What I found:
  - Dataset: 918 records, 12 features
  - [df.isnull](http://df.isnull?trk=public_post-text)().sum() → zero null values ✅
  - df.duplicated().sum() → 0 duplicates ✅
  - Cholesterol: min = 0 ❌ (medically impossible — 172 rows!)
  - Target: 508 positive, 410 negative — slightly imbalanced
  - Correlation heatmap: ST\_Slope\_Flat (0.55), ExerciseAngina\_Y (0.49), Oldpeak (0.39) — top predictors
  Data Cleaning & Preprocessing
  What I did:
  - 172 rows had Cholesterol = 0 → replaced with column mean (not dropped — that's 18% of data!)
  - Same fix for RestingBP zeros
  - pd.get\_dummies(drop\_first=True) → encoded Sex, ChestPainType, RestingECG, ExerciseAngina, ST\_Slope
  - StandardScaler on Age, RestingBP, Cholesterol, MaxHR, Oldpeak → Z-score normalization
  Feature Engineering & Selection
  What I did:
  - [pd.cut](http://pd.cut?trk=public_post-text)(Age) → created AgeGroup: Young / Middle / Senior / Old
  - pearsonr() → ranked all features by correlation with HeartDisease
  - chi2\_contingency() → Chi-Square test (alpha = 0.05) on all features
  Results:
  ✅ ST\_Slope\_Flat → Chi2=279.6, p=0.0 (Keep)
  ✅ ExerciseAngina\_Y → Chi2=222.3, p=0.0 (Keep)
  ✅ MaxHR → Chi2=241.3, p=0.0 (Keep)
  ❌ ChestPainType\_TA → Chi2=2.27, p=0.131 (Drop)
  ✅ Final dataset: 918 rows × 14 columns
  Same tools as Project 1. But this time a real data quality issue, a classification target, and stronger predictors.
  Stay tuned.
  [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#HeartDisease](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fheartdisease&trk=public_post-text) [#Classification](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fclassification&trk=public_post-text) [#EDA](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Feda&trk=public_post-text) [#FeatureEngineering](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffeatureengineering&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#pandas](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpandas&trk=public_post-text) [#sklearn](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsklearn&trk=public_post-text) [#LearningInPublic](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearninginpublic&trk=public_post-text) [#MLFromScratch](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmlfromscratch&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text) [#Beginners](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbeginners&trk=public_post-text) [#LearningJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearningjourney&trk=public_post-text)

  …more

  ![](https://media.licdn.com/dms/image/v2/D4D05AQFJ5_UGLIXt9Q/videocover-low/B4DZ5vSdQ0KABI-/0/1779983571688?e=2147483647&v=beta&t=fnb9l_FN0HnqJLvO9ANVBm28xG5kkkZ3iqtTsDppipI)Play Video

  Video Player is loading.

  Loaded: 0%

  PlayBack to start

  Stream Type LIVE

  Current Time 0:00

  /

  Duration 0:00

  1x

  Playback Rate

  Show Captions

  Mute

  Fullscreen



  [11](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashvilasmane_machinelearning-datascience-heartdisease-activity-7465792261243428865-5tty&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashvilasmane_machinelearning-datascience-heartdisease-activity-7465792261243428865-5tty&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashvilasmane_machinelearning-datascience-heartdisease-activity-7465792261243428865-5tty&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashvilasmane_machinelearning-datascience-heartdisease-activity-7465792261243428865-5tty&trk=public_post_feed-cta-banner-cta)
* [Monika J](https://in.linkedin.com/in/monika-j-882b611a3?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmonika-j-882b611a3_heart-diseases-preditction-activity-7466485939893112832-vb5n&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🚀 Excited to share my latest Machine Learning project: Heart Disease Prediction using Machine Learning ❤️🩺
  This project was an excellent opportunity to strengthen my understanding of the complete machine learning workflow, from data exploration to model evaluation.
  🔍 What I learned:
  ✅ Exploratory Data Analysis (EDA)
  Analyzed patient health data to identify patterns and trends.
  Explored relationships between features such as age, cholesterol levels, blood pressure, chest pain type, and heart disease occurrence.
  Used visualizations to uncover insights and understand data distribution.
  📊 Data Visualization
  Created graphs and charts to better interpret the dataset.
  Used correlation analysis to identify important features influencing predictions.
  Visualized patterns that helped guide model development.
  🤖 Model Building & Evaluation
  Trained machine learning models to predict the likelihood of heart disease.
  Evaluated model performance using key metrics:
  Accuracy
  Precision
  Recall
  F1 Score
  📈 Why these metrics matter
  Accuracy measures overall correctness.
  Precision evaluates how many positive predictions were actually correct.
  Recall measures the model's ability to identify true positive cases.
  F1 Score provides a balanced measure of Precision and Recall.
  💡 This project helped me understand that in healthcare applications, relying solely on accuracy is not enough. Metrics like Recall and F1 Score play a critical role in ensuring that potential heart disease cases are identified effectively.
  Through this project, I gained hands-on experience in:
  🔹 Data Cleaning
  🔹 Exploratory Data Analysis
  🔹 Data Visualization
  🔹 Machine Learning Model Development
  🔹 Model Performance Evaluation
  🔹 Healthcare Data Analytics
  Every project teaches something new, and this one significantly improved my understanding of how data-driven solutions can contribute to real-world healthcare challenges.
  A special thanks to our guide [Dr. Madderla Chiranjeevi](https://in.linkedin.com/in/chiru-madderla225?trk=public_post-text) [RV University](https://in.linkedin.com/school/rv-university/?trk=public_post-text)
  [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#HeartDiseasePrediction](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fheartdiseaseprediction&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#EDA](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Feda&trk=public_post-text) [#DataAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalytics&trk=public_post-text) [#HealthcareAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fhealthcareanalytics&trk=public_post-text) [#ArtificialIntelligence](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fartificialintelligence&trk=public_post-text) [#MachineLearningProject](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearningproject&trk=public_post-text) [#DataVisualization](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatavisualization&trk=public_post-text) [#ModelEvaluation](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmodelevaluation&trk=public_post-text) [#Accuracy](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Faccuracy&trk=public_post-text) [#Precision](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fprecision&trk=public_post-text) [#Recall](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Frecall&trk=public_post-text) [#F1Score](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ff1score&trk=public_post-text) [#LearningJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearningjourney&trk=public_post-text)




  [6](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmonika-j-882b611a3_heart-diseases-preditction-activity-7466485939893112832-vb5n&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmonika-j-882b611a3_heart-diseases-preditction-activity-7466485939893112832-vb5n&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmonika-j-882b611a3_heart-diseases-preditction-activity-7466485939893112832-vb5n&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmonika-j-882b611a3_heart-diseases-preditction-activity-7466485939893112832-vb5n&trk=public_post_feed-cta-banner-cta)
* [Yuan J.](https://www.linkedin.com/in/yuan-j-7505076?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyuan-j-7505076_biostatistics-bayesian-clinicaltrials-activity-7468033595807330304-7ZWt&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Bayesian thought of the day: There's a common worry that Bayesian methods don't belong in pivotal trials because they sneak in modelling assumptions a regulator shouldn't have to take on faith. I used to half-believe it too. But it mixes up two different things.
  The real divide in confirmatory inference isn't frequentist vs. Bayesian. It's design-based vs. model-based.
  What makes a randomised trial believable is the randomization. That's what justifies comparing the arms, without leaning on any assumed distribution for the outcomes. So the "modelling assumptions" worry is really a worry about analyses that claim more certainty than the randomisation can support, and a parametric frequentist test does that just as easily as a parametric Bayesian one.
  Here's the part that gets missed: Bayesian methods can also lean on the randomisation, not on a model. The Bayesian bootstrap and the finite-population (potential-outcomes) approach define the effect as the contrast among the actual randomised patients and let the randomisation carry the inference. The outcome model, if it's assumed, may increase efficiency (or not). That's about as few assumptions as the randomisation tests regulators already trust.
  So why bother with Bayes, if it has to play by the same rules?
  Because of what you get at the end. A posterior answers the question people actually ask: how likely is it this works, and by how much. A p-value answers something else, and we all know how often it gets misread. When you're weighing benefit against risk, a plain probability about the effect is just more useful than a tail probability under a null nobody believes.
  And the prior, done properly, is a feature. A pre-specified, skeptical prior puts your evidentiary bar out in the open where people can argue with it, instead of burying it in the choice of significance level and sample size. Show the result across a range of priors and everyone sees exactly how much skepticism the data can overcome.
  So you don't have to choose. You can build a pivotal-trial analysis on minimal modelling assumptions and still hand back the interpretable, decision-ready inference regulators actually need.
  The enemy was never priors. It was unjustified modelling assumptions, and you can dodge those in either language.
  [#Biostatistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbiostatistics&trk=public_post-text) [#Bayesian](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbayesian&trk=public_post-text) [#ClinicalTrials](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fclinicaltrials&trk=public_post-text) [#RegulatoryScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fregulatoryscience&trk=public_post-text)



  [101](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyuan-j-7505076_biostatistics-bayesian-clinicaltrials-activity-7468033595807330304-7ZWt&trk=public_post_social-actions-reactions)







  [35 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyuan-j-7505076_biostatistics-bayesian-clinicaltrials-activity-7468033595807330304-7ZWt&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyuan-j-7505076_biostatistics-bayesian-clinicaltrials-activity-7468033595807330304-7ZWt&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyuan-j-7505076_biostatistics-bayesian-clinicaltrials-activity-7468033595807330304-7ZWt&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyuan-j-7505076_biostatistics-bayesian-clinicaltrials-activity-7468033595807330304-7ZWt&trk=public_post_feed-cta-banner-cta)
* [Ifeoma James](https://ng.linkedin.com/in/ifeoma-james-4458321ba?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fifeoma-james-4458321ba_datascience-dataanalytics-descriptivestatistics-activity-7471204686583386112-19Cs&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  𝗪𝗵𝘆 𝘀𝘂𝗺𝗺𝗮𝗿𝘆 𝘀𝘁𝗮𝘁𝗶𝘀𝘁𝗶𝗰𝘀 𝘀𝗵𝗼𝘂𝗹𝗱 𝗮𝗹𝘄𝗮𝘆𝘀 𝗯𝗲 𝘆𝗼𝘂𝗿 𝗳𝗶𝗿𝘀𝘁 𝗿𝗲𝗮𝗹𝗶𝘁𝘆 𝗰𝗵𝗲𝗰𝗸 𝗶𝗻 𝗰𝗹𝗶𝗻𝗶𝗰𝗮𝗹 𝗮𝗻𝗮𝗹𝘆𝘁𝗶𝗰𝘀. 🩺📊
  Before building a machine learning model, you have to prove that your data actually makes sense. If your features don't align with real-world domain logic, your model's predictions won't either.
  While working on my project analyzing 100,000 patient records, I wanted to investigate the exact statistical relationship between a patient’s blood\_glucose\_level and their diabetes diagnosis.
  Instead of jumping straight to an algorithm, I used a fundamental data science tool: the .describe() function in Python, grouped directly by the diagnosis.
  The summary statistics revealed a massive, mathematically clear signal:
  🔹 The Healthy Cohort (Non-Diabetic):
  Mean Glucose: ~132.82 mg/dL
  Standard Deviation: Tight and controlled at ~34.06 mg/dL
  Clinical Context: This perfectly mirrors standard physiological baselines where fasting and post-meal glucose levels naturally cluster within a predictable, healthy range.
  🔹 The Diabetic Cohort:
  Mean Glucose: Skiered all the way up to 194.03 mg/dL
  Standard Deviation: Expanded drastically to 58.63 mg/dL
  Clinical Context: This massive spread reflects the high volatility and elevated baselines that characterize uncontrolled blood sugar in diabetic patients.
  𝗪𝗵𝘆 𝘁𝗵𝗶𝘀 𝘀𝘁𝗲𝗽 𝗶𝘀 𝗰𝗿𝘂𝗰𝗶𝗮𝗹 𝗳𝗼𝗿 𝗮 𝗗𝗮𝘁𝗮 𝗦𝗰𝗶𝗲𝗻𝘁𝗶𝘀𝘁:
  Running a grouped .describe() isn't just about looking at numbers on a screen. It’s about data validation.
  Seeing that the mean glucose for diabetic patients is significantly higher and far more variable tells me two things before I ever train a model:
  1️⃣ The dataset is clean and biologically accurate. It reflects true clinical reality.
  2️⃣ blood\_glucose\_level holds massive variance and will likely be one of the most powerful predictive features in my Random Forest classifier.
  To my fellow Scientists: What is your absolute favorite go-to descriptive function when you first open a completely raw, unverified dataset? Let’s share notes below! 👇
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#DataAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalytics&trk=public_post-text) [#DescriptiveStatistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdescriptivestatistics&trk=public_post-text) [#ExploratoryDataAnalysis](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fexploratorydataanalysis&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#Pandas](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpandas&trk=public_post-text) [#HealthcareAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fhealthcareanalytics&trk=public_post-text) [#DataValidation](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatavalidation&trk=public_post-text) [#TechCommunity](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ftechcommunity&trk=public_post-text)



  [6](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fifeoma-james-4458321ba_datascience-dataanalytics-descriptivestatistics-activity-7471204686583386112-19Cs&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fifeoma-james-4458321ba_datascience-dataanalytics-descriptivestatistics-activity-7471204686583386112-19Cs&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fifeoma-james-4458321ba_datascience-dataanalytics-descriptivestatistics-activity-7471204686583386112-19Cs&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fifeoma-james-4458321ba_datascience-dataanalytics-descriptivestatistics-activity-7471204686583386112-19Cs&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fifeoma-james-4458321ba_datascience-dataanalytics-descriptivestatistics-activity-7471204686583386112-19Cs&trk=public_post_feed-cta-banner-cta)
* [Priyanshu Kumar](https://in.linkedin.com/in/priyanshukumaar?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpriyanshukumaar_machinelearning-featureengineering-datascience-activity-7464986195073187840-XETH&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🧠 ML Day 28 — When simple imputation isn’t enough, let the data impute itself
  Today: Missing Indicators, Random Imputation, KNN Imputer, MICE
  Big realization:
  A missing value has two pieces of information:
  • the value itself is unknown
  • the fact that it’s missing may be predictive
  Missing Indicator
  Create a binary flag:
  feature\_missing = 1
  Then impute the actual value separately.
  Now the model learns:
  • the imputed value
  • the missingness pattern
  Random Sample Imputation
  Fill missing values by randomly sampling existing values.
  Advantage:
  • preserves original distribution
  • keeps variance intact
  Useful when simple mean imputation distorts the data too much.
  KNN Imputer
  Idea:
  similar rows should have similar values.
  For each missing value:
  • find nearest neighbours
  • impute using their average
  Works well when features are correlated.
  Important:
  KNN is distance-based →
  always scale features first.
  Iterative Imputer (MICE)
  Most statistically principled approach.
  Treat each missing feature as a prediction problem:
  • predict missing values using all other features
  • repeat iteratively until convergence
  Widely used in:
  • healthcare
  • finance
  • clinical research
  Practical hierarchy:
  • low missingness → mean/median
  • missingness carries signal → add indicators
  • correlated features → KNN / MICE
  • high-stakes domains → MICE preferred
  Most important insight today:
  Good imputation is not about filling blanks.
  It’s about preserving the underlying structure of the data.
  [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#FeatureEngineering](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffeatureengineering&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#ML](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fml&trk=public_post-text)




  [5](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpriyanshukumaar_machinelearning-featureengineering-datascience-activity-7464986195073187840-XETH&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpriyanshukumaar_machinelearning-featureengineering-datascience-activity-7464986195073187840-XETH&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpriyanshukumaar_machinelearning-featureengineering-datascience-activity-7464986195073187840-XETH&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpriyanshukumaar_machinelearning-featureengineering-datascience-activity-7464986195073187840-XETH&trk=public_post_feed-cta-banner-cta)
* [Madhav Kamble](https://in.linkedin.com/in/madhav-kamble-64710a221?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadhav-kamble-64710a221_%25F0%259D%2590%258D%25F0%259D%2590%25A8%25F0%259D%2590%25AD-%25F0%259D%2590%259A%25F0%259D%2590%25A5%25F0%259D%2590%25A5-%25F0%259D%2590%25A6%25F0%259D%2590%25A2%25F0%259D%2590%25AC%25F0%259D%2590%25AC%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%259D%25F0%259D%2590%259A%25F0%259D%2590%25AD%25F0%259D%2590%259A-%25F0%259D%2590%25A2%25F0%259D%2590%25AC-activity-7470572030556327936-sKjE&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  𝐍𝐨𝐭 𝐚𝐥𝐥 𝐦𝐢𝐬𝐬𝐢𝐧𝐠 𝐝𝐚𝐭𝐚 𝐢𝐬 𝐦𝐢𝐬𝐬𝐢𝐧𝐠 𝐟𝐨𝐫 𝐭𝐡𝐞 𝐬𝐚𝐦𝐞 𝐫𝐞𝐚𝐬𝐨𝐧.
  𝐓𝐡𝐚𝐭 𝐝𝐢𝐬𝐭𝐢𝐧𝐜𝐭𝐢𝐨𝐧 𝐦𝐚𝐭𝐭𝐞𝐫𝐬 𝐦𝐨𝐫𝐞 𝐭𝐡𝐚𝐧 𝐈 𝐞𝐱𝐩𝐞𝐜𝐭𝐞𝐝.
  While working on my MIGA research — a 𝙜𝙚𝙣𝙚𝙩𝙞𝙘 𝙖𝙡𝙜𝙤𝙧𝙞𝙩𝙝𝙢 approach to missing data imputation — I kept seeing three terms thrown around: MCAR, MAR, MNAR.
  I thought they were just academic labels.
  They're not.
  ■ 𝗠𝗖𝗔𝗥 — data is randomly absent.
  A sensor glitched. Someone skipped a field by accident. No deeper pattern. Imputation handles this reasonably well.
  ■ 𝗠𝗔𝗥 — missingness is linked to other variables you can observe.
  Younger patients skip follow-ups more often. You can model around it if you know what to look for.
  ■ 𝗠𝗡𝗔𝗥 — missingness is tied to the value that's missing itself.
  People with very high incomes tend not to report their income. The absence is the signal — but it won't tell you what it's hiding.
  𝘛𝘩𝘢𝘵 𝘭𝘢𝘴𝘵 𝘰𝘯𝘦 𝘪𝘴 𝘸𝘩𝘢𝘵 𝘣𝘳𝘦𝘢𝘬𝘴 𝘵𝘩𝘪𝘯g𝘴 𝘲𝘶𝘪𝘦𝘵𝘭𝘺.
  With MCAR and MAR, you're filling in blanks.
  With MNAR, you're trying to reconstruct something that disappeared because of what it was.
  No standard method handles this cleanly. Not MICE. Not KNN. Not mean imputation.
  Most benchmarks quietly skip it.
  I found that out the hard way when I added MNAR evaluation to my research and had to rethink how I was measuring imputation quality entirely.
  Missing data isn't just a preprocessing step.
  It's a research problem wearing a preprocessing costume.
  Have you ever run into MNAR in a real dataset? Curious how you handled it.



  [25](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadhav-kamble-64710a221_%25F0%259D%2590%258D%25F0%259D%2590%25A8%25F0%259D%2590%25AD-%25F0%259D%2590%259A%25F0%259D%2590%25A5%25F0%259D%2590%25A5-%25F0%259D%2590%25A6%25F0%259D%2590%25A2%25F0%259D%2590%25AC%25F0%259D%2590%25AC%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%259D%25F0%259D%2590%259A%25F0%259D%2590%25AD%25F0%259D%2590%259A-%25F0%259D%2590%25A2%25F0%259D%2590%25AC-activity-7470572030556327936-sKjE&trk=public_post_social-actions-reactions)







  [1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadhav-kamble-64710a221_%25F0%259D%2590%258D%25F0%259D%2590%25A8%25F0%259D%2590%25AD-%25F0%259D%2590%259A%25F0%259D%2590%25A5%25F0%259D%2590%25A5-%25F0%259D%2590%25A6%25F0%259D%2590%25A2%25F0%259D%2590%25AC%25F0%259D%2590%25AC%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%259D%25F0%259D%2590%259A%25F0%259D%2590%25AD%25F0%259D%2590%259A-%25F0%259D%2590%25A2%25F0%259D%2590%25AC-activity-7470572030556327936-sKjE&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadhav-kamble-64710a221_%25F0%259D%2590%258D%25F0%259D%2590%25A8%25F0%259D%2590%25AD-%25F0%259D%2590%259A%25F0%259D%2590%25A5%25F0%259D%2590%25A5-%25F0%259D%2590%25A6%25F0%259D%2590%25A2%25F0%259D%2590%25AC%25F0%259D%2590%25AC%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%259D%25F0%259D%2590%259A%25F0%259D%2590%25AD%25F0%259D%2590%259A-%25F0%259D%2590%25A2%25F0%259D%2590%25AC-activity-7470572030556327936-sKjE&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadhav-kamble-64710a221_%25F0%259D%2590%258D%25F0%259D%2590%25A8%25F0%259D%2590%25AD-%25F0%259D%2590%259A%25F0%259D%2590%25A5%25F0%259D%2590%25A5-%25F0%259D%2590%25A6%25F0%259D%2590%25A2%25F0%259D%2590%25AC%25F0%259D%2590%25AC%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%259D%25F0%259D%2590%259A%25F0%259D%2590%25AD%25F0%259D%2590%259A-%25F0%259D%2590%25A2%25F0%259D%2590%25AC-activity-7470572030556327936-sKjE&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadhav-kamble-64710a221_%25F0%259D%2590%258D%25F0%259D%2590%25A8%25F0%259D%2590%25AD-%25F0%259D%2590%259A%25F0%259D%2590%25A5%25F0%259D%2590%25A5-%25F0%259D%2590%25A6%25F0%259D%2590%25A2%25F0%259D%2590%25AC%25F0%259D%2590%25AC%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%259D%25F0%259D%2590%259A%25F0%259D%2590%25AD%25F0%259D%2590%259A-%25F0%259D%2590%25A2%25F0%259D%2590%25AC-activity-7470572030556327936-sKjE&trk=public_post_feed-cta-banner-cta)
* [Julius Bogomolovas](https://www.linkedin.com/in/julius-bogomolovas-93222a19?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjulius-bogomolovas-93222a19_conwaymaxwellbinomial-regression-two-directional-activity-7465072206511751169-dqUT&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  New experimental tool for "successes out of n" data: Conway–Maxwell–Binomial regression in glmmTMB.
  Binomial regression is often too rigid.
  Beta-binomial handles overdispersion, but only in one direction.
  Real bounded-count data can be overdispersed, underdispersed, or shift between the two across groups.
  CMB keeps the response between 0 and n while letting dispersion move on either side of the binomial case.
  I added a mean-parameterized CMB family to a fork of glmmTMB, with support for random effects and dispersion formulas. The mean parameterization is what makes it usable for regression.
  Where it could help: cell proliferation, fertilization assays, conversion rates, defaults, votes, and other "events out of opportunities" problems.
  Post + worked example:
  [https://lnkd.in/dkg95i5h](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fdkg95i5h&urlhash=-16R&trk=public_post-text)
  Very experimental. Try it, break it, and tell me where it fails.
  [#RStats](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Frstats&trk=public_post-text) [#Biostatistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbiostatistics&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#Econometrics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Feconometrics&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text)



  [6](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjulius-bogomolovas-93222a19_conwaymaxwellbinomial-regression-two-directional-activity-7465072206511751169-dqUT&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjulius-bogomolovas-93222a19_conwaymaxwellbinomial-regression-two-directional-activity-7465072206511751169-dqUT&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjulius-bogomolovas-93222a19_conwaymaxwellbinomial-regression-two-directional-activity-7465072206511751169-dqUT&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjulius-bogomolovas-93222a19_conwaymaxwellbinomial-regression-two-directional-activity-7465072206511751169-dqUT&trk=public_post_feed-cta-banner-cta)
* [SUJAN DHAKAL](https://np.linkedin.com/in/sujandhakal0?trk=public_post_feed-actor-name)

  2mo

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsujandhakal0_i-used-to-confuse-cross-validation-with-confusion-activity-7464584858837815298-DHi9&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  I used to confuse cross-validation with confusion matrix.
  When I am learning ML, I knew what cross-validation was. I knew what a confusion matrix and Sensitivity / Specificity  was.
  I couldn't tell you where each one actually fits when I'm building a model.
  Where do they fit in real model building?
  Let's say I'm predicting heart disease.
  First, I use cross-validation.
  I split my data into 5 folds. Train on 4, test on 1. Repeat 5 times.
  Why? So I don't get lucky with one random test set.
  For each test, I create a confusion matrix.
  It shows: true positive, false positive, true negative, false negative.
  Why? Accuracy hides the truth. The matrix shows exactly what my model messed up.
  Then I calculate sensitivity and specificity from that matrix.
  Sensitivity = did I catch the sick person?
  Specificity = did I falsely alarm a healthy person?
  Why? Because in heart disease, missing a sick person is worse than a false alarm.
  Before: Take your data and split it into two parts: training data (80%) and testing data (20%). Put the testing data away. Don't touch it until the very end.
  After: After cross-validation tells you which model is best (say Random Forest), train that same model again: but this time on all of the training data (the full 80%) , not just the 4 folds. Then test it once on that hidden 20% testing data. That final score is your real answer.
  the YouTube channel that helped me understand all the concepts most: [https://lnkd.in/gf5vxjx4](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fgf5vxjx4&urlhash=x9tv&trk=public_post-text) [StatQuest](https://www.linkedin.com/company/statquest?trk=public_post-text)



  [2](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsujandhakal0_i-used-to-confuse-cross-validation-with-confusion-activity-7464584858837815298-DHi9&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsujandhakal0_i-used-to-confuse-cross-validation-with-confusion-activity-7464584858837815298-DHi9&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsujandhakal0_i-used-to-confuse-cross-validation-with-confusion-activity-7464584858837815298-DHi9&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsujandhakal0_i-used-to-confuse-cross-validation-with-confusion-activity-7464584858837815298-DHi9&trk=public_post_feed-cta-banner-cta)
* [Statbitall](https://ca.linkedin.com/company/statbiall?trk=public_post_feed-actor-name)

  45 followers

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fstatbiall_the-central-limit-theorem-is-why-statistics-activity-7462125169545846784-lKns&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  The Central Limit Theorem is the most important result in statistics that most people can't explain past the name.
  Here's the core idea: take any population with any shape. Draw random samples and compute the mean of each one. As the sample size grows, the distribution of those sample means converges to a normal distribution, regardless of what the original population looked like.
  That single result is why confidence intervals have the formulas they do. It's why t-tests and z-tests work on real-world data that doesn't look remotely normal. It's the hidden foundation under most of the statistical machinery used in analytics and ML.
  But the theorem has conditions that are almost always glossed over.
  It requires independent observations, which breaks for time series and clustered data. It requires finite variance, which some financial return distributions don't have. And the sample size needed for the approximation to hold depends on how skewed the underlying population is — the common "n ≥ 30" rule is a rough guideline for symmetric distributions, not a universal law.
  This week's Statbitall post covers the theorem from its origins (de Moivre in 1733 through Lyapunov's 1901 proof) to a Python demonstration across three deliberately non-normal distributions, to the specific situations where it breaks down.
  If you've been invoking the CLT to justify normality assumptions without thinking about those conditions, this post is worth reading.
  [https://lnkd.in/eY3KSxUj](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FeY3KSxUj&urlhash=Z5Zk&trk=public_post-text)
  [#analytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalytics&trk=public_post-text) [#statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#data](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdata&trk=public_post-text) [#ai](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text) [#learning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearning&trk=public_post-text)



  [2](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fstatbiall_the-central-limit-theorem-is-why-statistics-activity-7462125169545846784-lKns&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fstatbiall_the-central-limit-theorem-is-why-statistics-activity-7462125169545846784-lKns&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fstatbiall_the-central-limit-theorem-is-why-statistics-activity-7462125169545846784-lKns&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fstatbiall_the-central-limit-theorem-is-why-statistics-activity-7462125169545846784-lKns&trk=public_post_feed-cta-banner-cta)

39,353 followers

* [1,620 Posts](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fin%2Fadrianolszewski%2Frecent-activity%2F&trk=public_post_follow-posts)
* [11 Articles](https://www.linkedin.com/today/author/adrianolszewski?trk=public_post_follow-articles)

[View Profile](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_follow-view-profile)
[Follow](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7461451559902830592&trk=public_post_follow)

## More from this author

* [### Type-M and Type-S errors in underpowered studies

  Adrian Olszewski

  1y](https://www.linkedin.com/pulse/type-m-type-s-errors-underpowered-studies-adrian-olszewski-c4mdf?trk=public_post)
* [### The role of the frequentist framework, NHST / p-values, statistical and practical significance in randomised controlled (clinical) trials

  Adrian Olszewski

  1y](https://www.linkedin.com/pulse/role-frequentist-framework-p-values-statistical-trials-olszewski-igy3f?trk=public_post)
* [### Importance of statistical significance and p-values in clinical trials

  Adrian Olszewski

  1y](https://www.linkedin.com/pulse/importance-statistical-significance-p-values-clinical-olszewski-mvyuf?trk=public_post)

## Explore content categories

* [Career](https://www.linkedin.com/top-content/career/)
* [Productivity](https://www.linkedin.com/top-content/productivity/)
* [Finance](https://www.linkedin.com/top-content/finance/)
* [Soft Skills & Emotional Intelligence](https://www.linkedin.com/top-content/soft-skills-emotional-intelligence/)
* [Project Management](https://www.linkedin.com/top-content/project-management/)
* [Education](https://www.linkedin.com/top-content/education/)
* [Technology](https://www.linkedin.com/top-content/technology/)
* [Leadership](https://www.linkedin.com/top-content/leadership/)
* [Ecommerce](https://www.linkedin.com/top-content/ecommerce/)
* [User Experience](https://www.linkedin.com/top-content/user-experience/)

Show more

Show less
