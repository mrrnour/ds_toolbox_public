---
url: https://www.linkedin.com/posts/adrianolszewski_please-dont-follow-this-panel-for-interpreting-share-7459858177883271168-5Ed4/?utm_source=share&utm_medium=member_android&rcm=ACoAACmphIUBo2D4cUQR_ZXSmdR7KMycDh8BUk8
scraped_at: 2026-07-27T14:10:13.611760
depth: 0
---

Please don't follow this panel for interpreting rank-based tests!
They test hypothesis of stochastic superiority, which is neither about means, medians, nor entire distributions in general. It means… | Adrian Olszewski



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# Adrian Olszewski’s Post

[Adrian Olszewski](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_feed-actor-name)

2mo

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

Please don't follow this panel for interpreting rank-based tests!
They test hypothesis of stochastic superiority, which is neither about means, medians, nor entire distributions in general. It means that unless strong distributional assumptions hold, the IID + symmetry around medians we are unable to say how two samples differ: by means? Medians? Variances? Skewness? Modes? It's very easy to obtain p>0.999 for very different means or medians and p<0.00..01 for equal means or medians even for small samples.
Stochastic superiority may be able to detect most kind of changes (in shape, dispersion, central tendency) but won't tell you what precisely. If you need to look at the details, consider quantile regression at different quantiles. This will tell how the data "behave" at different place of their distribution and will miss fewer details and keep the interpretation.
Rank-based methods are OK for ordinal data or if the nature of difference and the magnitudes can be ignored.
Also, permutation Welch t compares means without normality: [https://lnkd.in/dd2WHC2f](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fdd2WHC2f&urlhash=aiIZ&trk=public_post-text)
Check for references:
[https://lnkd.in/d\_wC-quu](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fd_wC-quu&urlhash=mE4q&trk=public_post-text)

[Sachin Nomula](https://in.linkedin.com/in/sachin-nomula?trk=public_post_reshare_feed-actor-name)

2mo

Choosing the right statistical test can completely change the accuracy of your analysis.
Here’s a simple roadmap to help you decide:
✔️ Compare 2 independent groups → Independent t-Test
✔️ Compare before vs after results → Paired t-Test
✔️ Compare 3+ groups → ANOVA
✔️ Non-normal data → Mann-Whitney / Kruskal-Wallis
✔️ Relationship between variables → Pearson / Spearman Correlation
✔️ Categorical variables → Chi-Square Test
✔️ Predict outcomes → Linear or Logistic Regression
Statistics becomes easier when you understand which test fits which problem.
A must-save guide for Data Science, Analytics, and Research enthusiasts.
Credits to [Venkata Naga Sai Kumar Bysani](https://www.linkedin.com/in/saibysani18?trk=public_post_reshare-text) for the insightful visual.
[#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post_reshare-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post_reshare-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post_reshare-text) [#Analytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalytics&trk=public_post_reshare-text) [#Research](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fresearch&trk=public_post_reshare-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post_reshare-text) [#Learning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearning&trk=public_post_reshare-text) [#DataAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalytics&trk=public_post_reshare-text) [#DataDrivenInsights](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatadriveninsights&trk=public_post_reshare-text)



[70](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_social-actions-reactions)







[7 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_social-actions-comments)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

[Hasib Ahmad](https://fi.linkedin.com/in/hasibahmad?trk=public_post_comment_actor-name)

2mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

I've been following you for a while and it seems you possess better understanding of stats than generic posts on LinkedIn. I would like to have your perspective or detail post on different tests and when they should be performed.

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_reply)
[1 Reaction](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_reactions)

2 Reactions

[Dr Mircea Zloteanu](https://uk.linkedin.com/in/mirceaz?trk=public_post_comment_actor-name)

2mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

People see the term stochastic difference and "leave the chat" (as the kids say). The issue is one that is field-wide; if you've never read a paper in your area to use this term or make inferences this way, a young researcher will say "why bother" and a senior one will say "it's been fine the old way, why change". I get this when I propose my changes - equivalence tests, effect size description, mixed effects models, non linear models. I just get looks like "ok nerd"

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_reply)
[3 Reactions](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_reactions)

4 Reactions

[Nassim AYAD](https://dz.linkedin.com/in/nassimayad?trk=public_post_comment_actor-name)

2mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

LLMs (chatgpt et al) generated charts 🤦♂️ 🤷♂️

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_reply)
[2 Reactions](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_comment_reactions)

3 Reactions

[See more comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_see-more-comments)

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_please-dont-follow-this-panel-for-interpreting-activity-7459858178747318273-DHVk&trk=public_post_feed-cta-banner-cta)

## More Relevant Posts

* [Aishwarya Puthran](https://ie.linkedin.com/in/aishwaryaputhran1?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faishwaryaputhran1_datascience-machinelearning-ai-activity-7466056458556702720-bSDO&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  One of the most useful lessons I have learned in Data Science:
  Before choosing a model, define the problem correctly.
  Most realworld use cases fall into these 4 categories.
  🔹 Classification → Predict categories
  Example: Will a customer churn?
  🔹 Regression → Predict numbers
  Example: What will sales be next year?
  🔹 Clustering → Discover hidden groups
  Example: How can we segment customers?
  🔹 Optimisation → Find the best decision
  Example: What is the best pricing strategy?
  As someone transitioning deeper into Data Science, I have realized that many realworld business challenges can be mapped back to these four foundations.
  Different problems → Different approaches → Better decisions
  What type of data science problem do you work on most often?
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text) [#Analytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalytics&trk=public_post-text) [#Data](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdata&trk=public_post-text) [#Learning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearning&trk=public_post-text) [#CareerGrowth](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcareergrowth&trk=public_post-text) [#DataScientist](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascientist&trk=public_post-text)

  + View C2PA information


  [3](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faishwaryaputhran1_datascience-machinelearning-ai-activity-7466056458556702720-bSDO&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faishwaryaputhran1_datascience-machinelearning-ai-activity-7466056458556702720-bSDO&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faishwaryaputhran1_datascience-machinelearning-ai-activity-7466056458556702720-bSDO&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faishwaryaputhran1_datascience-machinelearning-ai-activity-7466056458556702720-bSDO&trk=public_post_feed-cta-banner-cta)
* [Ashish Kumar Jha](https://in.linkedin.com/in/ashishkumarjha156483?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashishkumarjha156483_datascience-statistics-centrallimittheorem-activity-7465249195113881600-xFKo&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Most people memorize the Central Limit Theorem.
  But the real magic is visualizing how random samples slowly create a perfect bell curve 📊✨
  Statistics starts making sense when you can imagine the data moving.
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#CentralLimitTheorem](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcentrallimittheorem&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#Analytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalytics&trk=public_post-text) [#BusinessAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbusinessanalytics&trk=public_post-text) [#SQL](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsql&trk=public_post-text) [#DataAnalyst](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalyst&trk=public_post-text) [#Learning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearning&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text)

  + View C2PA information


  [5](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashishkumarjha156483_datascience-statistics-centrallimittheorem-activity-7465249195113881600-xFKo&trk=public_post_social-actions-reactions)







  [1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashishkumarjha156483_datascience-statistics-centrallimittheorem-activity-7465249195113881600-xFKo&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashishkumarjha156483_datascience-statistics-centrallimittheorem-activity-7465249195113881600-xFKo&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashishkumarjha156483_datascience-statistics-centrallimittheorem-activity-7465249195113881600-xFKo&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashishkumarjha156483_datascience-statistics-centrallimittheorem-activity-7465249195113881600-xFKo&trk=public_post_feed-cta-banner-cta)
* [Sai Ganesh M](https://in.linkedin.com/in/sai-ganesh-m-773208238?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-ganesh-m-773208238_datascience-analytics-machinelearning-activity-7460894255553679360-HcmP&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  💬 The most underrated skill in Data Science? Communication.
  You can build the best model.
  You can write perfect code.
  You can get amazing accuracy.
  But if you can’t explain the results, the project fails.
  Stakeholders don’t ask: “What algorithm did you use?”
  They ask:
  • What does this mean for the business?
  • What decision should we take?
  • How confident are we?
  • What is the risk?
  Data Scientists don’t just build models.
  They translate data into decisions.
  The real skill is turning numbers into stories 📊
  What skill do you think is most underrated in Data Science? 👇
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Analytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalytics&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#CareerGrowth](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcareergrowth&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text)



  [3](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-ganesh-m-773208238_datascience-analytics-machinelearning-activity-7460894255553679360-HcmP&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-ganesh-m-773208238_datascience-analytics-machinelearning-activity-7460894255553679360-HcmP&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-ganesh-m-773208238_datascience-analytics-machinelearning-activity-7460894255553679360-HcmP&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-ganesh-m-773208238_datascience-analytics-machinelearning-activity-7460894255553679360-HcmP&trk=public_post_feed-cta-banner-cta)
* [Data Science With Dennis | +1-775-242-6224](https://www.linkedin.com/company/dswithdennis?trk=public_post_feed-actor-name)

  942 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdswithdennis_machinelearning-datascience-ai-activity-7468447020148604929-E8iF&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Choosing between data science and statistics? Sometimes, a quick fix is better! Discover how to reduce workload from 10,000 to 500 records in just 1 month! Curious? Watch now!
  [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text) [#Tech](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ftech&trk=public_post-text) [#CareerGrowth](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcareergrowth&trk=public_post-text)

  …more

  ![](https://media.licdn.com/dms/image/v2/D5610AQGJgArhLvMWrA/videocover-high/B56Z6VA7t3GgAo-/0/1780616519176?e=2147483647&v=beta&t=y1PcMI8EfRXxKolmA7FjZnLaKwWiSc0y9AOBqQhnSAE)Play Video

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



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdswithdennis_machinelearning-datascience-ai-activity-7468447020148604929-E8iF&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdswithdennis_machinelearning-datascience-ai-activity-7468447020148604929-E8iF&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdswithdennis_machinelearning-datascience-ai-activity-7468447020148604929-E8iF&trk=public_post_feed-cta-banner-cta)
* [Ashraf R.](https://sg.linkedin.com/in/ashrafrahim83?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashrafrahim83_option-2-the-controlled-burn-activity-7462490602333442048-KgTe&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  ### 🌀 Option 2: The Controlled Burn
  \*\*Mohammad Ashraf Bin Abdul Rahim\*\*
  \*Data professional with the calm energy of someone who's debugged a dashboard at 2 AM and lived to tell the tale.
  Background: Culinary arts → Financial Analytics → AI. Yes, it's a plot twist.
  Certified in deep learning, DAX, and knowing when to close the 47 browser tabs.
  Currently: Building smarter data solutions | Seeking teams that value curiosity over "culture fit."
  \*Not a founder. Just really good at making numbers behave.\*
  this is what comes out when i ask AI to write an unhinged bio about me.



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashrafrahim83_option-2-the-controlled-burn-activity-7462490602333442048-KgTe&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashrafrahim83_option-2-the-controlled-burn-activity-7462490602333442048-KgTe&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashrafrahim83_option-2-the-controlled-burn-activity-7462490602333442048-KgTe&trk=public_post_feed-cta-banner-cta)
* [Claude Beazley](https://ch.linkedin.com/in/claude-beazley-99777993?trk=public_post_feed-actor-name)

  1mo

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fclaude-beazley-99777993_actually-the-second-fruity-set-of-the-activity-7470215341890113536-Yu1U&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Actually the second , fruity, set of the equations is not a clear representation of the original model equations.
  Although there are 2 bunches of bananas, there are 4 bananas in each bunch. without the context of the first set of equations, we have no way of knowing whether the fruity equations are referring to the total number of bananas or just bunches of bananas.
  And while it seems a bit silly to be so pedantic about bananas, what if that was a population dynamics model? Those bunches could be entire cohorts.
  So while it is very important to represent data in a more digestable form, be careful not to create unnecessary ambiguity.

  [José Vargas](https://sv.linkedin.com/in/josevargasdev?trk=public_post_reshare_feed-actor-name)

  Confidencial en Empresa Confidencial

  1mo

  Your stakeholders rarely need to see a mathematical model first. They need to understand what decision they can make based on its result.
  This meme illustrates it very well.
  The equation is correct: if 3x = 30, then x = 10. And if x + 2y = 18, then y = 4. The mathematical logic works.
  But the way you present it can completely change the conversation.
  When we show only formulas, part of the audience has to spend energy translating the analysis before they can discuss its business implications. In a business meeting, that extra effort can create friction.
  When we bring the same logic into a clear visualization, the message becomes more accessible. The apples and bananas could represent revenue, retention, churn, CAC, margin, or any critical business variable.
  That is where the value of Data Storytelling appears.
  A good technical analysis explains what is happening. A good data story helps people understand why it matters, what risks exist, and what action should be taken.
  For a Data Scientist, the work should not stop at making the model predict accurately. It should also help the organization understand the result clearly enough to act on it.
  The right visualization does not weaken the rigor. It makes the logic behind the analysis visible.
  Because a prediction that no one understands rarely becomes a decision.
  How do you usually translate technical findings into clear messages for business audiences?
  by [José Vargas](https://sv.linkedin.com/in/josevargasdev?trk=public_post_reshare-text)
  [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post_reshare-text) [#datascience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post_reshare-text)



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fclaude-beazley-99777993_actually-the-second-fruity-set-of-the-activity-7470215341890113536-Yu1U&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fclaude-beazley-99777993_actually-the-second-fruity-set-of-the-activity-7470215341890113536-Yu1U&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fclaude-beazley-99777993_actually-the-second-fruity-set-of-the-activity-7470215341890113536-Yu1U&trk=public_post_feed-cta-banner-cta)
* [Nobert Wafula](https://ke.linkedin.com/in/nobert-wafula?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnobert-wafula_data-modelling-activity-7460624521062981632-vAX3&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Before carrying out any data modeling or statistical analysis, it is important to ensure that your data meets the necessary assumptions. Unfortunately, many analysts overlook this critical step, which can lead to misleading results and poor decision-making.
  Some key assumptions to always check include:
  • Normality
  • Outliers
  • Numeric Variables
  • Linear Relationships
  • Equal Variance
  • Independent Observations
  • Homoscedasticity
  In this document, I have explored the methods used to test each assumption, how to determine whether a model satisfies them, and the most appropriate solutions to apply when violations occur.
  Data analysis is not just about running models or generating visualizations it is about understanding the data, handling it correctly, and making evidence-based decisions that are reliable and meaningful.
  Strong models begin with strong data preparation.
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#DataAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalytics&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#RStats](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Frstats&trk=public_post-text) [#DataModeling](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatamodeling&trk=public_post-text) [#DataAnalysis](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalysis&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text) [#BusinessIntelligence](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbusinessintelligence&trk=public_post-text) [#Analytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalytics&trk=public_post-text)




  [20](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnobert-wafula_data-modelling-activity-7460624521062981632-vAX3&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnobert-wafula_data-modelling-activity-7460624521062981632-vAX3&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnobert-wafula_data-modelling-activity-7460624521062981632-vAX3&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnobert-wafula_data-modelling-activity-7460624521062981632-vAX3&trk=public_post_feed-cta-banner-cta)
* [Ayan Ganguly](https://in.linkedin.com/in/ayan-ganguly-b87386297?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fayan-ganguly-b87386297_machinelearning-datascience-python-activity-7467596879019782144-te9J&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🚀 End-to-End Machine Learning Project: From Prediction to Classification
  Recently, I completed a project combining both Regression and Classification techniques to solve real-world problems using data.
  📊 Part A: Medical Cost Prediction (Regression)
  Built a Linear Regression model to predict insurance charges.
  🔍 Key Insights:
  • Charges are right-skewed with high-cost outliers
  • Smoking is the most impactful factor (huge increase in charges)
  • Age and BMI show moderate positive influence
  💡 Takeaway:
  Lifestyle factors (like smoking) dominate medical cost prediction more than basic demographics.
  🎬 Part B: Movie Success Prediction (Classification)
  Predicted whether a movie is a Hit (rating ≥ 7) or Flop using Logistic Regression.
  🔍 Key Insights:
  • Dataset is imbalanced (~79% Flop vs ~21% Hit)
  • Popularity & vote count strongly influence success
  • Budget alone does not guarantee a hit
  📈 Learning:
  • Accuracy is not enough → F1-score & Recall matter more in imbalanced data
  ⚙️ What I Learned Overall:
  • Importance of EDA in understanding data patterns
  • Handling missing values and imbalance
  • Difference between regression vs classification problems
  • Why simple models are great for baseline but not always production-ready
  💡 Final Thought:
  Good models don’t just predict — they help us understand the story behind the data.
  Project Notebook:
  [[https://lnkd.in/gVTfkU5X](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FgVTfkU5X&urlhash=EQcj&trk=public_post-text)]
  [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#Regression](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fregression&trk=public_post-text) [#Classification](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fclassification&trk=public_post-text) [#StudentProject](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstudentproject&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text) [#LearningJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearningjourney&trk=public_post-text)



  [1](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fayan-ganguly-b87386297_machinelearning-datascience-python-activity-7467596879019782144-te9J&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fayan-ganguly-b87386297_machinelearning-datascience-python-activity-7467596879019782144-te9J&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fayan-ganguly-b87386297_machinelearning-datascience-python-activity-7467596879019782144-te9J&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fayan-ganguly-b87386297_machinelearning-datascience-python-activity-7467596879019782144-te9J&trk=public_post_feed-cta-banner-cta)
* [Ajay jestin](https://ae.linkedin.com/in/ajayjestin-ds?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fajayjestin-ds_datascience-statistics-machinelearning-activity-7465763935854616576-tSPG&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Statistics is not just theory — it plays a huge role in real-world Data Science and business decision-making 📊
  Understanding concepts like Quartiles, IQR, Skewness, Covariance, Correlation, Random Variables, and Set Theory helps solve real-time data problems across industries.
  🔹 Quartiles & IQR
  Used to detect outliers in employee salaries, fraud transactions, sensor failures, and abnormal business data.
  🔹 Skewness
  Helps understand customer behavior, sales trends, and income distribution where data is not evenly distributed.
  🔹 Correlation & Covariance
  Used to analyze relationships between variables like:
  • Sales vs Marketing Spend
  • Experience vs Salary
  • Temperature vs Energy Consumption
  🔹 Random Variables
  Important in probability-based systems such as forecasting, risk analysis, recommendation systems, and Machine Learning models.
  🔹 Set Theory
  Used in databases, SQL operations, segmentation, and recommendation engines.
  These concepts are the foundation behind data cleaning, EDA, Machine Learning, dashboards, and predictive analytics.
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#DataAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalytics&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text) [#BusinessAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbusinessanalytics&trk=public_post-text) [#LearningJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearningjourney&trk=public_post-text) [#FutureDataScientist](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffuturedatascientist&trk=public_post-text)



  [2](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fajayjestin-ds_datascience-statistics-machinelearning-activity-7465763935854616576-tSPG&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fajayjestin-ds_datascience-statistics-machinelearning-activity-7465763935854616576-tSPG&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fajayjestin-ds_datascience-statistics-machinelearning-activity-7465763935854616576-tSPG&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fajayjestin-ds_datascience-statistics-machinelearning-activity-7465763935854616576-tSPG&trk=public_post_feed-cta-banner-cta)
* [Sai Deepthi P S G](https://www.linkedin.com/in/sai-deepthi-g-p-s?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-deepthi-g-p-s_datascience-machinelearning-analytics-activity-7459973247049273344-LtFD&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Data Science Mistakes That Matter — Part 1: Correlation Isn’t Causation
  Your model didn’t find a signal.
  It found a coincidence.
  This is one of the easiest mistakes to make in data science.
  Two things move together… so we assume one must be causing the other.
  But that’s not always true.
  A classic example:
  🍦 Ice cream sales go up
  🦈 Shark attacks also go up
  At first glance, it looks related.
  But ice cream doesn’t cause shark attacks.
  👉 Hot weather drives both.
  That’s correlation without causation.
  I’ve seen similar situations in real business problems too.
  For example:
  You notice that customers who use premium support churn less.
  So the conclusion becomes:
  👉 “Let’s push more users to premium support to reduce churn.”
  Sounds reasonable.
  But what if loyal customers are naturally more likely to use premium support in the first place?
  Now the relationship changes:
  • Premium support didn’t reduce churn
  • Loyalty is driving both
  The model picked up a pattern — but not the real cause.
  And this is where decisions can go wrong.
  Because if we act on correlation alone:
  • we invest in the wrong solution
  • the model still looks “right”
  • but business outcomes don’t improve
  One thing that helped me think about this better:
  👉 Not every useful feature is causal — but most business decisions assume causality.
  Quick takeaway:
  Just because two things move together doesn’t mean one causes the other.
  Always ask:
  👉 What’s the real driver behind this pattern?
  Curious — have you seen examples where correlation led teams in the wrong direction?
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#Analytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalytics&trk=public_post-text) [#AI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fai&trk=public_post-text) [#CausalInference](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcausalinference&trk=public_post-text) [#DataScienceCommunity](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatasciencecommunity&trk=public_post-text)

  + View C2PA information


  [11](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-deepthi-g-p-s_datascience-machinelearning-analytics-activity-7459973247049273344-LtFD&trk=public_post_social-actions-reactions)







  [6 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-deepthi-g-p-s_datascience-machinelearning-analytics-activity-7459973247049273344-LtFD&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-deepthi-g-p-s_datascience-machinelearning-analytics-activity-7459973247049273344-LtFD&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-deepthi-g-p-s_datascience-machinelearning-analytics-activity-7459973247049273344-LtFD&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsai-deepthi-g-p-s_datascience-machinelearning-analytics-activity-7459973247049273344-LtFD&trk=public_post_feed-cta-banner-cta)

39,353 followers

* [1,620 Posts](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fin%2Fadrianolszewski%2Frecent-activity%2F&trk=public_post_follow-posts)
* [11 Articles](https://www.linkedin.com/today/author/adrianolszewski?trk=public_post_follow-articles)

[View Profile](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_follow-view-profile)
[Follow](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7459858178747318273&trk=public_post_follow)

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
