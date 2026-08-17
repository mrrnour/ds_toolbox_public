---
url: https://www.linkedin.com/posts/adrianolszewski_statistics-datascience-normality-share-7294020157369192448-G3ON/?utm_source=share&utm_medium=member_android
scraped_at: 2026-07-27T14:10:19.982146
depth: 0
---

Let me make a few notes about normality tests: | Adrian Olszewski



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# Adrian Olszewski’s Post

[Adrian Olszewski](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_feed-actor-name)

1y

Edited

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

Let me make a few notes about normality tests:
1) Jarque-Bera - a weak one as based only the highest moments. This way it can miss deep holes (or multiple modes) if only skewness and kurtosis is kept. I always advise against it. Same iss has the Cullen-Frey plot, determining distributions in the skewness-kurtosis space.
2) Kolmogorov-Smirnov (and its improved resamplig version - Lilliefors) looks only at one point - the maximum difference. Cramer von Misses is the same idea but integrated over the whole support. Anderson-Darling weights CvM to be more sensitive at tails, thus superior
3) Shapiro-Wilk is the weakest from the whole Shapiro-xx family (still almost the strongest ex aequo with AD) but seems to be the fastest, so probably that's why it's so popular. In some cases surprisingly it was weaker than Geary's test
4) bearing in mind these 40+ tests are split into families, each sensitive to something else, it's not surprising the can can totally contradict each other.
5) These tests do NOT tell you what kind of deviation from the theoretical normality occurred 👉 One should always remember this and always follow them with QQ plots.
📌 more remarks about the "normality tests" - what does it mean to test for "normality" (and why it's wrong to say that! it should be AGAINST normality)
[https://lnkd.in/dSGfqj3i](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FdSGfqj3i&urlhash=-zKf&trk=public_post-text) and [https://lnkd.in/dVNEE4Zh](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FdVNEE4Zh&urlhash=lx7L&trk=public_post-text)
📌 40+ non-normality tests - why do we have so many tests and can they actually contradict each other?
[https://lnkd.in/deJyWWub](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FdeJyWWub&urlhash=OCfl&trk=public_post-text)
📌 Why some people find them useless (I don't - but it's always good to know both sides!)? A discussion:
[https://lnkd.in/dnTeFpT8](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FdnTeFpT8&urlhash=TYG3&trk=public_post-text)
[#statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#datascience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#normality](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fnormality&trk=public_post-text) [#data](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdata&trk=public_post-text)




[119](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_social-actions-reactions)







[4 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_social-actions-comments)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

[Shyam Joshi, PhD, PE](https://www.linkedin.com/in/shyam-joshi-mathworks?trk=public_post_comment_actor-name)

1y

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

[Adrian Olszewski](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_comment-text) the hypothesis to test must be decided a-priori. In theory, it’s not ok to visualize data and then decide hypothesis to test! And there are 40+ hypothesis tests. So how many and which Normality hypothesis tests do you recommend. I like adtest and kstest. My favorite is Filiben’s probability plot correlation coefficient test of normality.

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment_reply)
[2 Reactions](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment_reactions)

3 Reactions

[Giorgio Pioda](https://ch.linkedin.com/in/piodag?trk=public_post_comment_actor-name)

1y

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

The worst part of the problem is that frequently the interest is in NOT rejecting the H0.
Thus, QQ plot (& simulated data overlay) is the best choice.

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment_reply)
[3 Reactions](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_comment_reactions)

4 Reactions

[See more comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_see-more-comments)

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_statistics-datascience-normality-activity-7294020158572945408-HjDQ&trk=public_post_feed-cta-banner-cta)

## More Relevant Posts

* [Robin Beura](https://www.linkedin.com/in/robin-beura?trk=public_post_feed-actor-name)

  9mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frobin-beura_covariance-vs-correlation-the-misunderstood-activity-7383473284769595392-BBOB&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  We often use covariance and correlation interchangeably — but they’re not twins, they’re siblings with very different personalities.
  Covariance tells you if two variables move together.
  Correlation tells you how strongly and how consistently they move together — and that tiny distinction shapes how we interpret everything from A/B tests to marketing lift to model performance.
  In this new article, I unpack how these two metrics quietly power causal reasoning in analytics — and why understanding their limits can save you from misleading conclusions (especially when running experiments or building models).
  👉 Read it here:
  [https://lnkd.in/g66vUrfC](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fg66vUrfC&urlhash=Srk-&trk=public_post-text)
  It’s a short, concept-first read — perfect if you want to reconnect with the math that drives modern experimentation frameworks.
  [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#Experimentation](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fexperimentation&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Analytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalytics&trk=public_post-text) [#Causality](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcausality&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text)



  [3](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frobin-beura_covariance-vs-correlation-the-misunderstood-activity-7383473284769595392-BBOB&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frobin-beura_covariance-vs-correlation-the-misunderstood-activity-7383473284769595392-BBOB&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frobin-beura_covariance-vs-correlation-the-misunderstood-activity-7383473284769595392-BBOB&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frobin-beura_covariance-vs-correlation-the-misunderstood-activity-7383473284769595392-BBOB&trk=public_post_feed-cta-banner-cta)
* [Aparna P](https://in.linkedin.com/in/itsaparnap?trk=public_post_feed-actor-name)

  10mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fitsaparnap_datascience-deeplearning-rnn-activity-7378077378981519360-U5tR&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Weather Temperature Prediction with RNN
  Built a Recurrent Neural Network (SimpleRNN) to forecast next-day temperatures using historical weather data.
  Workflow
  Preprocessed daily weather dataset → handled missing values, normalized features
  Created input sequences (past 7–14 days)
  Implemented SimpleRNN with 32–64 units + Dense output
  Trained & validated using MSE loss, Adam optimizer, and MAE as evaluation metric
  Results
  Evaluated with RMSE, MAE, and R²
  Achieved reliable short-term forecasting
  Visualized predicted vs. actual temperatures + 7-day future forecast
  This project enhanced my skills in time-series modeling, RNNs, and weather data forecasting.
  Github : "[https://lnkd.in/g7jVDbFy](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fg7jVDbFy&urlhash=tf76&trk=public_post-text)"
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#DeepLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdeeplearning&trk=public_post-text) [#RNN](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Frnn&trk=public_post-text) [#TimeSeries](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ftimeseries&trk=public_post-text) [#WeatherForecasting](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fweatherforecasting&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text)



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fitsaparnap_datascience-deeplearning-rnn-activity-7378077378981519360-U5tR&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fitsaparnap_datascience-deeplearning-rnn-activity-7378077378981519360-U5tR&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fitsaparnap_datascience-deeplearning-rnn-activity-7378077378981519360-U5tR&trk=public_post_feed-cta-banner-cta)
* [Thiyanga Talagala](https://au.linkedin.com/in/thiyanga-talagala-24bb63134?trk=public_post_feed-actor-name)

  9mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthiyanga-talagala-24bb63134_datascience-r-anomalydetection-activity-7387089858273505280-oUAI&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  New CRAN Task View: Anomaly Detection 🚨
  Are you struggling to find the right R packages for anomaly detection?
  Don’t worry — the CRAN Task View on Anomaly Detection by [Priyanga Dilini Talagala](https://lk.linkedin.com/in/priyanga-dilini-talagala-47b269122?trk=public_post-text), Rob J. Hyndman, and Gaetano Romano is now live on CRAN! 🎉
  Check it at: [https://lnkd.in/g6WVUYds](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fg6WVUYds&urlhash=U0mT&trk=public_post-text)
  This Task View provides a comprehensive and curated list of R packages for detecting anomalies, outliers, novelties, and extreme values across different data types — including univariate, multivariate, temporal, spatial, spatio-temporal, and functional data. [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#R](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fr&trk=public_post-text) [#AnomalyDetection](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanomalydetection&trk=public_post-text)




  [92](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthiyanga-talagala-24bb63134_datascience-r-anomalydetection-activity-7387089858273505280-oUAI&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthiyanga-talagala-24bb63134_datascience-r-anomalydetection-activity-7387089858273505280-oUAI&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthiyanga-talagala-24bb63134_datascience-r-anomalydetection-activity-7387089858273505280-oUAI&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthiyanga-talagala-24bb63134_datascience-r-anomalydetection-activity-7387089858273505280-oUAI&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthiyanga-talagala-24bb63134_datascience-r-anomalydetection-activity-7387089858273505280-oUAI&trk=public_post_feed-cta-banner-cta)
* [Sinan Süha Tepebaşılı](https://tr.linkedin.com/in/sinantepebasili?trk=public_post_feed-actor-name)

  9mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsinantepebasili_modelinterpretability-featureimportance-activity-7380883733802868737-cZWK&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Permutation Importance sheds new light on model interpretability, highlighting a crucial aspect for those diving into this realm.
  It challenges the conventional wisdom of traditional feature importance metrics like Gini in Random Forests, which can lead astray, particularly in the presence of correlated or unevenly scaled features.
  The stark drop in R² from 0.82 to 0.65 post shuffling Temperature speaks volumes. The model's integrity hinges on these signals, showcasing the pivotal role of each feature.
  As you aptly put it, "A feature’s value isn’t just in how much it’s used — it’s in what happens when it’s gone." This resonates deeply with experiences where seemingly minor features, like "weekday vs. weekend" in a traffic model, unexpectedly unravel significant congestion patterns.
  Exploring SHAP values unveils another layer of insight, often reshaping our understanding of feature impact. It's a journey that constantly reshapes our perspectives on model dynamics.
  [#ModelInterpretability](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmodelinterpretability&trk=public_post-text) [#FeatureImportance](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffeatureimportance&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text)

  + View C2PA information


  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsinantepebasili_modelinterpretability-featureimportance-activity-7380883733802868737-cZWK&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsinantepebasili_modelinterpretability-featureimportance-activity-7380883733802868737-cZWK&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsinantepebasili_modelinterpretability-featureimportance-activity-7380883733802868737-cZWK&trk=public_post_feed-cta-banner-cta)
* [Anurag Potdar](https://in.linkedin.com/in/anuragpotdar?trk=public_post_feed-actor-name)

  9mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fanuragpotdar_100daysofml-100daysofml-machinelearning-activity-7381667198714417153-aOAB&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Day 22 [#100DaysOfML](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2F100daysofml&trk=public_post-text)
  Polynomial Regression
  Not all relationships in data are straight lines. Sometimes, the curve tells a better story and that’s where Polynomial Regression comes in.
  Polynomial Regression extends Linear Regression by adding higher-order terms of the input variable allowing the model to capture non-linear patterns.
  🔹 Concept:
  We start with the linear equation:
  [ y = β₀ + β₁x + ε ]
  Then introduce polynomial terms to make it nonlinear:
  [ y = β₀ + β₁x + β₂x² + β₃x³ + … + ε ]
  Even though the relationship between x and y becomes nonlinear, the model remains linear in parameters that’s why it’s still called Polynomial Linear Regression.
  🔹 Why use it?
  Fits curves and complex patterns
  Great for datasets where trends aren’t linear
  Often used in economics, physics, or growth modeling
  🔹 Visualize it:
  Plot your data points and fit curves of different polynomial degrees —
  you’ll see how increasing the degree bends the line to fit more complex shapes.
  Takeaway
  Polynomial Regression lets your model see beyond straight lines
  but remember, more flexibility can also mean more overfitting!
  [#100DaysOfML](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2F100daysofml&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#Regression](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fregression&trk=public_post-text) [#PolynomialRegression](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpolynomialregression&trk=public_post-text) [#CurveFitting](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcurvefitting&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#LearningJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearningjourney&trk=public_post-text)



  [4](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fanuragpotdar_100daysofml-100daysofml-machinelearning-activity-7381667198714417153-aOAB&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fanuragpotdar_100daysofml-100daysofml-machinelearning-activity-7381667198714417153-aOAB&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fanuragpotdar_100daysofml-100daysofml-machinelearning-activity-7381667198714417153-aOAB&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fanuragpotdar_100daysofml-100daysofml-machinelearning-activity-7381667198714417153-aOAB&trk=public_post_feed-cta-banner-cta)
* [Yashraj Shrivastava](https://in.linkedin.com/in/yashrajshrivastava17?trk=public_post_feed-actor-name)

  9mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashrajshrivastava17_chi-square-test-anova-types-of-anova-p-value-activity-7387448136186785792-LgEm&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  📊 Inferential Statistics — Demystified.
  Statistics isn’t just about numbers; it’s about inference — drawing meaningful conclusions from data.
  In our latest NeuralMinds episode, we dive into one of the most applied yet misunderstood areas of Data Science: Inferential Statistics (Part 2) — covering Chi-Square Tests, ANOVA, its types, and the story behind the p-value.
  🎯 Here’s what you’ll gain from this session:
  -->Clear intuition behind Chi-Square and ANOVA tests
  -->Understanding when and why to use each
  -->The role of p-values in decision-making
  -->Real-world examples connecting theory to Data Science practice
  This content is designed for learners who believe that understanding why matters more than just knowing how.
  Special thanks to [Krish Naik](https://in.linkedin.com/in/naikkrish?trk=public_post-text) Sir for his constant mentorship and guidance — helping us build clarity and curiosity in every step of learning. 🙏
  [#NeuralMinds](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fneuralminds&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#ANOVA](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanova&trk=public_post-text) [#ChiSquare](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fchisquare&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#InferentialStatistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Finferentialstatistics&trk=public_post-text) [#KrishNaik](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fkrishnaik&trk=public_post-text)

  …more

  [### Chi-Square Test, ANOVA, Types of ANOVA, p-Value Explained | Inferential Statistics Part 2](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Fwww%2Eyoutube%2Ecom%2Fwatch%3Fv%3DLqMNcMetuJs&urlhash=V8o7&trk=public_post_ingested-content-summary-external-video-content)

  #### https://www.youtube.com/



  [8](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashrajshrivastava17_chi-square-test-anova-types-of-anova-p-value-activity-7387448136186785792-LgEm&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashrajshrivastava17_chi-square-test-anova-types-of-anova-p-value-activity-7387448136186785792-LgEm&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashrajshrivastava17_chi-square-test-anova-types-of-anova-p-value-activity-7387448136186785792-LgEm&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fyashrajshrivastava17_chi-square-test-anova-types-of-anova-p-value-activity-7387448136186785792-LgEm&trk=public_post_feed-cta-banner-cta)
* [Dr. Ammar HOMAIDA](https://tr.linkedin.com/in/ammar-homaida?trk=public_post_feed-actor-name)

  9mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fammar-homaida_liutyperegression-appliedstatistics-fuzzymodels-activity-7380583793889366016-yXGC&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  The Overlooked Hero — Liu-Type Regression
  While OLS, Ridge, and Lasso get most of the attention, Liu-type regression offers a flexible alternative that merges Ridge’s stability and Liu’s bias correction.
  🔹 It introduces two tuning parameters (k, d), giving researchers greater control over bias–variance trade-offs.
  🔹 With careful selection, it often yields superior predictive accuracy.
  For complex or fuzzy datasets, Liu-type regression can quietly outperform the classics.
  [#LiuTypeRegression](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fliutyperegression&trk=public_post-text) [#AppliedStatistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fappliedstatistics&trk=public_post-text) [#FuzzyModels](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffuzzymodels&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Regression](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fregression&trk=public_post-text)



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fammar-homaida_liutyperegression-appliedstatistics-fuzzymodels-activity-7380583793889366016-yXGC&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fammar-homaida_liutyperegression-appliedstatistics-fuzzymodels-activity-7380583793889366016-yXGC&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fammar-homaida_liutyperegression-appliedstatistics-fuzzymodels-activity-7380583793889366016-yXGC&trk=public_post_feed-cta-banner-cta)
* [Spatial and Data Science Society of Nigeria](https://ng.linkedin.com/company/spatial-and-data-science-society?trk=public_post_feed-actor-name)

  3,712 followers

  9mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fspatial-and-data-science-society_bayesianmodeling-spatialanalysis-datascience-activity-7381322109513396224-pqej&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🌍 Bayesian Spatial Models: Turning Uncertainty into Insight
  Uncertainty is not the enemy it’s part of the story. Bayesian spatial models go beyond traditional analysis, capturing uncertainty to strengthen how we interpret patterns and make decisions. From environmental risks to urban systems, they help transform raw data into reliable, evidence-based insights.
  [#BayesianModeling](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbayesianmodeling&trk=public_post-text) [#SpatialAnalysis](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fspatialanalysis&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Uncertainty](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Funcertainty&trk=public_post-text) [#DecisionMaking](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdecisionmaking&trk=public_post-text) [#SDSSN](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsdssn&trk=public_post-text)




  [14](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fspatial-and-data-science-society_bayesianmodeling-spatialanalysis-datascience-activity-7381322109513396224-pqej&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fspatial-and-data-science-society_bayesianmodeling-spatialanalysis-datascience-activity-7381322109513396224-pqej&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fspatial-and-data-science-society_bayesianmodeling-spatialanalysis-datascience-activity-7381322109513396224-pqej&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fspatial-and-data-science-society_bayesianmodeling-spatialanalysis-datascience-activity-7381322109513396224-pqej&trk=public_post_feed-cta-banner-cta)
* [Adrian Olszewski](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_feed-actor-name)

  9mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-anova-activity-7387137673657745408-wxdx&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  In the past I talked about non-parametric approaches to ANOVA ➡️ ART-ANOVA, ANOVA-Type Statistic (ATS), Wald-Type Statistic (WTS), and a few more methods. But sometimes you also need to adjust your analysis for numerical covariates, needing something like a "non-parametric ANCOVA". Below I show a few examples made in R. Hope it helps!
  ⚠️ But remember - and never forget - "non-parametric" ≠ "no problems"! Actually, one set of problems with distributional assumptions turns into a new set of problems: interpretation.
  ⚠️ And one more thing: "non-parametric AN[C]OVA" is NOT an AN[C]OVA anymore if based on ranks, quantiles or anything else that doesn't refer to means and the analysis of reduction in the residual variance. If you also like Box-Cox transforming your response, you compares something even more weird.
  / 2 exceptions: are distribution-free ANOVA's that preserve the original hypothesis: permutation ANOVA, ANOVA over a GEE-estimated linear model /
  In other words, you answer now a ⚠️ different hypothesis, and whether it's consistent with your original one is a different question you must be able to answer. For those, who can answer such question, I show a few ways of doing something in the spirit of non-parametric "ANCOVA"
  / 💡 Whatever approach you take, the core concept is similar: an analysis of how levels of a set of categorical factors jointly affect the response. It doesn't matter what type of data and measure it is - medians, hazards, probabilities, ranks, means, whatever.
  It goes like this:
  1) analysis of the main effects of some model via joint Wald's tests of orthogonal contrasts,
  2) if the model belongs to the Generalized Linear Model family it becomes an analysis of deviance, done via Wald's, Likelihood Ration or Rao score test,
  3) if the model is a special case of the GLM - a general linear (Gaussian) model, deviance becomes residual variance and it's about the reduction of residual variance ==> ANOVA, with exact F statistic (=LRT = Wald = Rao). /
  [#statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#datascience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#anova](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanova&trk=public_post-text) [#ancova](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fancova&trk=public_post-text) [#research](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fresearch&trk=public_post-text) [#nonparametric](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fnonparametric&trk=public_post-text)




  [54](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-anova-activity-7387137673657745408-wxdx&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-anova-activity-7387137673657745408-wxdx&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-anova-activity-7387137673657745408-wxdx&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-anova-activity-7387137673657745408-wxdx&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_statistics-datascience-anova-activity-7387137673657745408-wxdx&trk=public_post_feed-cta-banner-cta)
* [NoiseGrasp](https://cl.linkedin.com/company/noisegrasp?trk=public_post_feed-actor-name)

  401 followers

  10mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnoisegrasp_bayesian-mmm-marketingscience-activity-7378768081525055488-xwzy&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  In theory, you can wait for “Big Data.” In reality, most teams can’t.
  Frequentist methods often shine with large samples, but day to day constraints are messy: sparse histories, fragmented channels, shifting budgets.
  A Bayesian approach encode prior knowledge, borrow strength across segments and tune parameters carefully to produce reliable models from imperfect data.
  [#Bayesian](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbayesian&trk=public_post-text) [#MMM](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmmm&trk=public_post-text) [#MarketingScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmarketingscience&trk=public_post-text) [#DecisionScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdecisionscience&trk=public_post-text) [#NoiseGrasp](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fnoisegrasp&trk=public_post-text)




  [6](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnoisegrasp_bayesian-mmm-marketingscience-activity-7378768081525055488-xwzy&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnoisegrasp_bayesian-mmm-marketingscience-activity-7378768081525055488-xwzy&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnoisegrasp_bayesian-mmm-marketingscience-activity-7378768081525055488-xwzy&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnoisegrasp_bayesian-mmm-marketingscience-activity-7378768081525055488-xwzy&trk=public_post_feed-cta-banner-cta)

39,353 followers

* [1,620 Posts](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fin%2Fadrianolszewski%2Frecent-activity%2F&trk=public_post_follow-posts)
* [11 Articles](https://www.linkedin.com/today/author/adrianolszewski?trk=public_post_follow-articles)

[View Profile](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_follow-view-profile)
[Connect](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7294020158572945408&trk=public_post_follow)

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
