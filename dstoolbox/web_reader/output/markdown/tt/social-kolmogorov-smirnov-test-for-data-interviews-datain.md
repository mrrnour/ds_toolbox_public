---
url: https://www.linkedin.com/posts/what-is-the-kolmogorov-smirnov-test-in-share-7470142858088513537-TfxE/?utm_source=share&utm_medium=member_android&rcm=ACoAACmphIUBo2D4cUQR_ZXSmdR7KMycDh8BUk8
scraped_at: 2026-07-27T14:12:47.417993
depth: 0
---

Kolmogorov-Smirnov Test for Data Interviews | DataInterview.com posted on the topic | LinkedIn



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# Kolmogorov-Smirnov Test for Data Interviews

This title was summarized by AI from the post below.

[DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_feed-actor-name)

26,554 followers

1mo

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

What is the Kolmogorov-Smirnov Test? (in Data interviews)
👋 Let's learn together ↓
The KS Test is a 𝗻𝗼𝗻𝗽𝗮𝗿𝗮𝗺𝗲𝘁𝗿𝗶𝗰 𝘁𝗲𝘀𝘁 𝘁𝗵𝗮𝘁 𝗰𝗼𝗺𝗽𝗮𝗿𝗲𝘀 𝘁𝘄𝗼 𝗱𝗶𝘀𝘁𝗿𝗶𝗯𝘂𝘁𝗶𝗼𝗻𝘀 𝗯𝘆 𝗳𝗶𝗻𝗱𝗶𝗻𝗴 𝘁𝗵𝗲 𝗹𝗮𝗿𝗴𝗲𝘀𝘁 𝗴𝗮𝗽 𝗯𝗲𝘁𝘄𝗲𝗲𝗻 𝘁𝗵𝗲𝗶𝗿 𝗖𝗗𝗙𝘀.
No assumption about the underlying distribution. That's the whole point. You just need two samples and you can ask: do these come from the same distribution?
In ML, this shows up constantly. Training data vs. serving data. Last week vs. this week. If the gap is big enough, something shifted.
📐 𝗧𝗵𝗲 𝘁𝗲𝘀𝘁 𝘀𝘁𝗮𝘁𝗶𝘀𝘁𝗶𝗰:
D(n,m) = sup over x of |Fn(x) - Gm(x)|
Where:
Fn(x) → empirical CDF of sample A (step function, jumps by 1/n at each observation)
Gm(x) → empirical CDF of sample B
sup → supremum, the largest vertical gap across all x values
D → the test statistic. Bigger D means more evidence the distributions differ.
Critical value at 5%: D\_α = 1.36 × sqrt((n+m) / nm)
Reject H₀ when D > D\_α.
⚡ 𝗛𝗼𝘄 𝗶𝘁 𝘄𝗼𝗿𝗸𝘀:
① Compute the empirical CDF for each sample
② At every observed value, calculate |Fn(x) - Gm(x)|
③ Find the maximum gap D across all those points
④ Compare D to the critical value for your chosen alpha
⑤ If D exceeds the threshold, reject the null (distributions are the same)
🧐 𝗛𝗼𝘄 𝗶𝘀 𝗶𝘁 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁 𝗳𝗿𝗼𝗺 𝘁𝗵𝗲 𝗖𝗵𝗶-𝗦𝗾𝘂𝗮𝗿𝗲 𝗚𝗼𝗼𝗱𝗻𝗲𝘀𝘀-𝗼𝗳-𝗙𝗶𝘁 𝗧𝗲𝘀𝘁?
KS works on continuous data and needs no binning. Chi-Square requires you to bucket data into bins, and the result depends on how you do it.
KS is sensitive to differences anywhere in the distribution. Chi-Square can miss localized shifts depending on bin choice.
KS is distribution-free by default. Chi-Square assumes enough expected counts per bin.
KS is weaker in the tails and against heavy-tail scale shifts. Chi-Square can sometimes catch those better with the right binning.
✍️ 𝗪𝗵𝗲𝗻 𝘁𝗼 𝘂𝘀𝗲 𝘁𝗵𝗲 𝗞𝗦 𝗧𝗲𝘀𝘁:
when you need to detect data drift in ML pipelines, run two-sample A/B tests on continuous metrics, or check if a feature distribution has shifted between time periods, without assuming any specific distribution shape.
👉 Land Data & AI jobs on [datainterview.com](https://www.linkedin.com/redir/redirect?url=http%3A%2F%2Fdatainterview%2Ecom&urlhash=bCYW&trk=public_post-text)




[133](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_social-actions-reactions)







[2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_social-actions-comments)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

[Mohammed Moniruzzaman Khan](https://pt.linkedin.com/in/mohammed-moniruzzaman-khan?trk=public_post_comment_actor-name)

1mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

This is a helpful breakdown. I’ve also seen KS test used in model monitoring pipelines for detecting dataset shift, especially when comparing simulated vs real data in stochastic models.

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_comment_reply)

1 Reaction

[DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_comment_actor-name)

1mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

👉 Land data, AI, quant jobs on [datainterview.com](http://datainterview.com?trk=public_post_comment-text)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_comment_reply)

1 Reaction

[See more comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_see-more-comments)

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_feed-cta-banner-cta)

## More Relevant Posts

* [DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_feed-actor-name)

  26,554 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-mcnemars-test-in-data-interviews-activity-7471592301224599553-8Lgg&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  What is McNemar's Test? (in Data interviews)
  👋 Let's learn together ↓
  McNemar's Test is a 𝗽𝗮𝗶𝗿𝗲𝗱 𝗻𝗼𝗻-𝗽𝗮𝗿𝗮𝗺𝗲𝘁𝗿𝗶𝗰 𝘁𝗲𝘀𝘁 𝗳𝗼𝗿 𝗰𝗼𝗺𝗽𝗮𝗿𝗶𝗻𝗴 𝘁𝘄𝗼 𝗰𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗲𝗿𝘀 𝗼𝗻 𝘁𝗵𝗲 𝘀𝗮𝗺𝗲 𝘁𝗲𝘀𝘁 𝘀𝗲𝘁.
  The key idea: accuracy difference alone doesn't tell you much. What matters is the disagreements. Cases where model A is right and B is wrong, and vice versa. Those are the only pairs that carry signal.
  Cases where both models agree (both right or both wrong) cancel out completely. You're testing whether one model systematically beats the other on the hard cases.
  📐 𝗧𝗵𝗲 𝘁𝗲𝘀𝘁 𝘀𝘁𝗮𝘁𝗶𝘀𝘁𝗶𝗰:
  χ² = (b - c)² / (b + c) ~ χ²₁
  Where:
  b → cases where only Model A is correct
  c → cases where only Model B is correct
  b + c → total discordant pairs (the only ones that matter)
  χ²₁ → chi-squared with 1 degree of freedom
  Reject H₀ when χ² > 3.84 at α = 0.05.
  For small samples (b + c < 25), skip the chi-squared form. Use the exact binomial instead: p = 2 × sum from k=0 to min(b,c) of C(b+c, k) × (0.5)^(b+c)
  💪 𝗛𝗼𝘄 𝘁𝗼 𝗿𝘂𝗻 𝗶𝘁:
  ① Build a 2×2 contingency table of paired predictions
  ② Fill cells: a (both correct), b (only A correct), c (only B correct), d (both wrong)
  ③ Ignore a and d. Focus only on b and c
  ④ If b + c ≥ 25, compute χ² = (b - c)² / (b + c)
  ⑤ If b + c < 25, use exact binomial p-value
  🧐 𝗛𝗼𝘄 𝗶𝘀 𝗶𝘁 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁 𝗳𝗿𝗼𝗺 𝗮 𝟮-𝘀𝗮𝗺𝗽𝗹𝗲 𝘇-𝘁𝗲𝘀𝘁 𝗼𝗻 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆?
  A z-test compares overall accuracy rates and treats the two models as independent samples.
  McNemar's uses the paired structure. Same test items, same conditions. That pairing removes a lot of noise.
  McNemar's is more statistically powerful when models are evaluated on the same data. The z-test can miss real differences. McNemar's won't.
  Also: for more than 2 models, use Cochran's Q instead.
  ✍️ 𝗪𝗵𝗲𝗻 𝘁𝗼 𝘂𝘀𝗲 𝗠𝗰𝗡𝗲𝗺𝗮𝗿'𝘀 𝗧𝗲𝘀𝘁:
  when you want to know if two classifiers differ significantly on the same held-out set, and raw accuracy comparison isn't enough.
  👉 Land Data & AI jobs on [datainterview.com](https://www.linkedin.com/redir/redirect?url=http%3A%2F%2Fdatainterview%2Ecom&urlhash=bCYW&trk=public_post-text)




  [75](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-mcnemars-test-in-data-interviews-activity-7471592301224599553-8Lgg&trk=public_post_social-actions-reactions)







  [1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-mcnemars-test-in-data-interviews-activity-7471592301224599553-8Lgg&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-mcnemars-test-in-data-interviews-activity-7471592301224599553-8Lgg&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-mcnemars-test-in-data-interviews-activity-7471592301224599553-8Lgg&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-mcnemars-test-in-data-interviews-activity-7471592301224599553-8Lgg&trk=public_post_feed-cta-banner-cta)
* [Jamilla Cooiman](https://nl.linkedin.com/in/jamilla-cooiman-5076a8248?trk=public_post_feed-actor-name)

  3w

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjamilla-cooiman-5076a8248_causal-forests-wager-athey-paper-activity-7478089371787767809-Wb3h&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  In an A/B test, we are often interested not only in the average effect of a treatment, but also in how that effect varies across units.
  For example, suppose we run an A/B test to estimate the effect of a new onboarding flow on activation. The average difference in activation rates tells us whether the new flow improves activation on average, but in practice we may also want to know whether the effect differs across user characteristics, such as acquisition channel, device type, region, or prior product experience.
  That kind of heterogeneity can matter a lot, because it can inform targeting decisions.
  One way to explore this is to use a regression model with treatment interactions. For example, we could interact the treatment indicator with acquisition channel, device type, region, and so forth.
  However, this creates a practical problem: how do we choose which heterogeneity dimensions to explore?
  Usually, we can think of many variables over which effect heterogeneity could exist, but exploring all of them is not feasible.
  The more granularly we slice the data, the smaller the effective sample sizes become. Uncertainty increases and if we search across enough subgroup differences, some patterns will start to appear just by chance.
  So in practice, we need to be careful about which heterogeneity we explore and how we explore it.
  One approach that can help here, under the right conditions, is to use causal forests.
  To put it simply, a causal forest is similar to a random forest, but it is built for a different target.
  A standard random forest tries to find splits that improve outcome prediction. More specifically, it tries to create leaves where the outcomes are relatively homogeneous, so that the model can predict the outcome well.
  A causal forest instead tries to find splits that are informative about effect heterogeneity. More specifically, each tree tries to split the data into leaves where the difference between treated and control units varies across leaves, so that the different leaves correspond to regions of the covariate space with meaningfully different effects.
  In that sense, causal forests can reduce some of the manual subgroup-searching burden, because they search for heterogeneity in a data-driven way.
  At the same time, the procedure also considers the statistical side of the problem. It does not only reward large apparent differences in treatment effects; it also uses safeguards such as minimum leaf sizes, enough treated and control observations within leaves, and honesty, where different samples are used to choose the tree splits and to estimate the effects inside the final leaves.
  Of course, causal forests do not remove the need for causal identification assumptions, but when the causal design is credible, I think they are a promising method for studying treatment effect heterogeneity in business settings.
  For those looking to go deeper, the paper below is one of the foundational works in this area.




  [106](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjamilla-cooiman-5076a8248_causal-forests-wager-athey-paper-activity-7478089371787767809-Wb3h&trk=public_post_social-actions-reactions)







  [4 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjamilla-cooiman-5076a8248_causal-forests-wager-athey-paper-activity-7478089371787767809-Wb3h&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjamilla-cooiman-5076a8248_causal-forests-wager-athey-paper-activity-7478089371787767809-Wb3h&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjamilla-cooiman-5076a8248_causal-forests-wager-athey-paper-activity-7478089371787767809-Wb3h&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjamilla-cooiman-5076a8248_causal-forests-wager-athey-paper-activity-7478089371787767809-Wb3h&trk=public_post_feed-cta-banner-cta)
* [Shubham Parihar](https://in.linkedin.com/in/shubhamparihar7?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshubhamparihar7_knowing-about-your-%25F0%259D%2597%2597%25F0%259D%2597%25AE%25F0%259D%2598%2581%25F0%259D%2597%25AE%25F0%259D%2598%2580-%25F0%259D%2597%2596%25F0%259D%2597%25B5%25F0%259D%2597%25AE%25F0%259D%2597%25BF%25F0%259D%2597%25AE-activity-7472890406385422336-vj3p&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Knowing about your 𝗗𝗮𝘁𝗮'𝘀 𝗖𝗵𝗮𝗿𝗮𝗰𝘁𝗲𝗿𝗶𝘀𝘁𝗶𝗰𝘀 is Ultra Important before doing any further analysis...
  While working on my Data Analysis Project , yesterday I got to learn a very important lesson.
  This is so Important that I can guarantee if a data professional makes mess in this step then all the further analytical results will be of no use.
  So I had framed a Diagnostic Question for my Internal Stakeholder.
  𝗤 ➔ Is app size associated with install volume ?
  I had to find the answer of this and share the insight to the stakeholder.
  In order to answer this I used two columns "Size\_In\_MB" and "Installs".
  Next step was to Visualize the Relationship using a Scatter Plot.
  Honestly the visualization was actually tricky for me to see it and conclude that what type of correlation exists.
  And that is where I had learnt another lesson which was backing a correlation visualization with a "Correlation Coefficient".
  Sometimes visualization's are hard to interprete and thats where Correlation Coefficient help us by providing the Numerical Value.
  Anyways coming back to the main topic.
  I have already identified columns for finding answer and also Plotted a Scatter Plot.
  Now its time to find Correlation Coefficient.
  There are many methods for calculation of Correlation Coefficient out of which "Pearson" and "Spearman" are the most commonly used.
  And this is the step ( Choosing Correlation Coefficient Method ) where a big mistake can happen.
  The big mistake is "𝗜𝗴𝗻𝗼𝗿𝗶𝗻𝗴 𝗬𝗼𝘂𝗿 𝗗𝗮𝘁𝗮'𝘀 𝗖𝗵𝗮𝗿𝗮𝗰𝘁𝗲𝗿𝗶𝘀𝘁𝗶𝗰𝘀".
  Why ? So listen carefully.
  Every different method has its own Assumptions , works well with certain type of data , has certain needs and more...
  If that doesnt matches with your data , The result can be totally wrong.
  𝗣𝗲𝗮𝗿𝘀𝗼𝗻 needs :-
  ➫ Both variables to be numerical.
  ➫ Linear relationship.
  ➫ Sensitive to outliers.
  𝗦𝗽𝗲𝗮𝗿𝗺𝗮𝗻 needs :-
  ➫ Numerical , ordinal , or ranked data
  ➫ Monotonic relationship.
  ➫ Less sensitive to outliers.
  Now I already knew about my Data's Characteristics very well...
  ➢ Size\_In\_MB was continuous , Installs was Ordinal Categorical.
  ➢ Installs had many outliers.
  ➢ Size\_In\_MB was positively skewed data.
  Hence I had Implemented "Spearman's Correlation" which gave me the Coefficient as "0.31" meaning a "Weak Correlation".
  And then when I Implemented "Pearson's Correlation" it gave me the Coefficient as "0.13" meaning a "Very Weak Correlation".
  Just see the difference , according to...
  Spearman ➟ Weak Correlation.
  Pearson ➟ Very Weak Correlation.
  That is why choosing a wrong method will lead to wrong analytical results and interpretations.
  Therfore it is very important to know first your "Data's Characteristics".
  The lesson which I learnt was "It is ultra important to know firstly your Data's Characteristics , Secondly knowing Assumptions and Requirements of a Method and then given correct information of both the things Employ the Suitable Method".




  [2](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshubhamparihar7_knowing-about-your-%25F0%259D%2597%2597%25F0%259D%2597%25AE%25F0%259D%2598%2581%25F0%259D%2597%25AE%25F0%259D%2598%2580-%25F0%259D%2597%2596%25F0%259D%2597%25B5%25F0%259D%2597%25AE%25F0%259D%2597%25BF%25F0%259D%2597%25AE-activity-7472890406385422336-vj3p&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshubhamparihar7_knowing-about-your-%25F0%259D%2597%2597%25F0%259D%2597%25AE%25F0%259D%2598%2581%25F0%259D%2597%25AE%25F0%259D%2598%2580-%25F0%259D%2597%2596%25F0%259D%2597%25B5%25F0%259D%2597%25AE%25F0%259D%2597%25BF%25F0%259D%2597%25AE-activity-7472890406385422336-vj3p&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshubhamparihar7_knowing-about-your-%25F0%259D%2597%2597%25F0%259D%2597%25AE%25F0%259D%2598%2581%25F0%259D%2597%25AE%25F0%259D%2598%2580-%25F0%259D%2597%2596%25F0%259D%2597%25B5%25F0%259D%2597%25AE%25F0%259D%2597%25BF%25F0%259D%2597%25AE-activity-7472890406385422336-vj3p&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshubhamparihar7_knowing-about-your-%25F0%259D%2597%2597%25F0%259D%2597%25AE%25F0%259D%2598%2581%25F0%259D%2597%25AE%25F0%259D%2598%2580-%25F0%259D%2597%2596%25F0%259D%2597%25B5%25F0%259D%2597%25AE%25F0%259D%2597%25BF%25F0%259D%2597%25AE-activity-7472890406385422336-vj3p&trk=public_post_feed-cta-banner-cta)
* [UDDEISHYA KUMAR](https://in.linkedin.com/in/uddeishya-kumar-130283253?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fuddeishya-kumar-130283253_practical-data-science-series-day-1348-activity-7472997087463403521-sgYK&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Practical Data Science Series Day (13/48):
  "A 95% Confidence Interval means there's a 95% chance the true value is inside it."
  True or False?
  If you said True - you're in good company. Most people do.
  But the answer is False.
  And getting this wrong isn't trivial. It's one of the most deeply rooted misconceptions in statistics - and it quietly drives bad decisions in boardrooms every day.
  The marksman in the dark.
  Picture a shooter firing at a target he can't see.
  He fires 100 rounds.
  95 land within a 10cm ring around the bullseye.
  5 miss completely.
  Lights go off again.
  He fires shot 101.
  Is there a 95% probability this shot hit the ring?
  No.
  That shot has already landed. It's either inside or it isn't.
  The 95% was never about this one shot. It described his method - his process of aiming and firing.
  Over many shots, that process hits the target about 95% of the time.
  That's your confidence interval
  The true value you're measuring - conversion rate, revenue, treatment effect - is fixed.
  It doesn't move.
  Your interval either captured it, or it missed. You just can't know which
  So what does 95% actually mean?
  If you ran the same study 100 times and built 100 intervals - roughly 95 would contain the true value.
  And 5 would miss entirely
  The interval you're holding right now? Could be one of the 95. Could be one of the 5. No way to tell from inside the result
  Common misconceptions I hear constantly:
  → "We are 95% sure conversion rate is between 4.2% and 6.8%."
  Wrong framing. Your procedure is right 95% of the time across repetitions. This specific interval has already either captured the truth or missed it
  → "Wider interval means more certainty."
  No. Wider means less precise, less informative - not more confident
  → "Overlapping intervals mean no significant difference."
  Also false. Overlapping intervals can still yield a significant difference
  Test it directly
  Where this gets dangerous
  Your team tells the board: "We're 95% confident the campaign drove a 12% lift." Budget approved
  But your interval was one of the 5% that missed. The true lift was zero. Now the company is scaling something that never worked
  Not because anyone lied. Because everyone confused the reliability of the method with certainty about this result.
  Key takeaway:
  A 95% CI is a statement about your procedure across many repetitions - not a probability statement about this one interval
  Your interval either contains the truth, or it doesn't. You simply cannot know which.
  The interval moves. The truth never does.
  ~~
  Content Credits: [Aditya Rai](https://in.linkedin.com/in/aditya-rai-ds?trk=public_post-text)
  ♻️ Repost if you want your network to stop misinterpreting confidence intervals
  📌 Follow me ([UDDEISHYA KUMAR](https://in.linkedin.com/in/uddeishya-kumar-130283253?trk=public_post-text)) for more data science and statistics content




  [9](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fuddeishya-kumar-130283253_practical-data-science-series-day-1348-activity-7472997087463403521-sgYK&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fuddeishya-kumar-130283253_practical-data-science-series-day-1348-activity-7472997087463403521-sgYK&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fuddeishya-kumar-130283253_practical-data-science-series-day-1348-activity-7472997087463403521-sgYK&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fuddeishya-kumar-130283253_practical-data-science-series-day-1348-activity-7472997087463403521-sgYK&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fuddeishya-kumar-130283253_practical-data-science-series-day-1348-activity-7472997087463403521-sgYK&trk=public_post_feed-cta-banner-cta)
* [Greptime | The Single Database for Observability on S3](https://www.linkedin.com/company/greptime?trk=public_post_feed-actor-name)

  1,512 followers

  3w

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgreptime_observability-sre-databaseengineering-activity-7478480058115035136-l3dN&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Observability has a version number now. Most teams are still on 1.0 without realizing it.
  The three-pillar model — metrics, logs, traces — has been the default for nearly a decade. It works. But it was designed around a constraint that no longer exists: storage was expensive, so you pre-aggregated everything before writing it down.
  𝗢𝗯𝘀𝗲𝗿𝘃𝗮𝗯𝗶𝗹𝗶𝘁𝘆 𝟭.𝟬 𝗺𝗮𝗸𝗲𝘀 𝗮 𝘄𝗿𝗶𝘁𝗲-𝘁𝗶𝗺𝗲 𝗯𝗲𝘁 𝘆𝗼𝘂 𝗰𝗮𝗻'𝘁 𝘂𝗻𝗱𝗼.
  When you instrument a service, you decide upfront which dimensions your metrics carry. High-cardinality fields like 𝘣𝘶𝘪𝘭𝘥\_𝘪𝘥, 𝘳𝘦𝘨𝘪𝘰𝘯, or 𝘱𝘢𝘺𝘮𝘦𝘯𝘵\_𝘱𝘳𝘰𝘷𝘪𝘥𝘦𝘳 are often excluded at instrumentation time because they'd blow up cardinality. Logs keep the text but drop the schema. Traces are often sampled. Each pillar discards something at write time, and you can't get it back at 2am when you need it.
  𝗢𝗯𝘀𝗲𝗿𝘃𝗮𝗯𝗶𝗹𝗶𝘁𝘆 𝟮.𝟬 𝗳𝗹𝗶𝗽𝘀 𝘁𝗵𝗲 𝗺𝗼𝗱𝗲𝗹.
  Instead of pre-aggregating at write time, you emit one 𝘄𝗶𝗱𝗲 𝗲𝘃𝗲𝗻𝘁 per request or span with every field attached. Metrics, trace views, and log views become different queries over the same raw data, derived at read time. You can GROUP BY 𝘣𝘶𝘪𝘭𝘥\_𝘪𝘥 after the incident, not just before it. Charity Majors at Honeycomb formalized this as "wide events as a single source of truth." The idea isn't new — Meta's Scuba was doing it in 2013. What changed is that columnar storage on object storage finally made it affordable.
  𝗔𝗜 𝗮𝗴𝗲𝗻𝘁𝘀 𝗮𝗿𝗲 𝘄𝗵𝗲𝗿𝗲 𝘁𝗵𝗲 𝘂𝗽𝗴𝗿𝗮𝗱𝗲 𝗯𝗲𝗰𝗼𝗺𝗲𝘀 𝘂𝗿𝗴𝗲𝗻𝘁.
  A single agent execution step carries model name, token counts, the full prompt, tool call parameters, reasoning, and memory state — 50 to 200 fields per event. The questions you need to ask aren't "is it up" or "what's the p99." They're "was the answer accurate," "did it hallucinate," "why did it pick that tool." Those are questions you can only answer by keeping the raw event. Honeycomb's data shows mature observability datasets typically carry 𝟭𝟬𝟬+ dimensions per event. Pre-aggregation can't cover that range.
  We've been working through these trade-offs while building GreptimeDB, an open-source database that stores metrics, logs, and traces in a single engine on object storage. Three posts covering the full reasoning are up:
  👉🏻 Wide Events, Explained: The Data Model Behind Observability 2.0: [https://lnkd.in/gcE4yAUE](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FgcE4yAUE&urlhash=ZAyP&trk=public_post-text)
  👉🏻 Agent Observability: [https://lnkd.in/gW7QSAN2](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FgW7QSAN2&urlhash=JSQ6&trk=public_post-text)
  👉🏻 Database for Observability 2.0: [https://lnkd.in/gSJEm7EY](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FgSJEm7EY&urlhash=rLLU&trk=public_post-text)
  Drop a comment if you're thinking through this shift in your own stack.
  [#Observability](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fobservability&trk=public_post-text) [#SRE](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsre&trk=public_post-text) [#DatabaseEngineering](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatabaseengineering&trk=public_post-text) [#OpenSource](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fopensource&trk=public_post-text) [#AIAgents](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Faiagents&trk=public_post-text)




  [1](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgreptime_observability-sre-databaseengineering-activity-7478480058115035136-l3dN&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgreptime_observability-sre-databaseengineering-activity-7478480058115035136-l3dN&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgreptime_observability-sre-databaseengineering-activity-7478480058115035136-l3dN&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgreptime_observability-sre-databaseengineering-activity-7478480058115035136-l3dN&trk=public_post_feed-cta-banner-cta)
* [Srinija Velaga](https://in.linkedin.com/in/srinija-velaga-42312b1b4?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsrinija-velaga-42312b1b4_dotnet-aspnetcore-webapi-activity-7473794379581816832-Zbm0&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🚀 Day 72 — Filtering & Sorting in APIs (Helping users find exactly what they need)
  Let’s start with a real scenario 👇
  ❓ Your products database contains:
  ✔️ 500,000 Products
  A client wants:
  \* Only Electronics
  \* Price less than ₹50,000
  \* Sorted by Rating
  Should they download all products and filter them locally? 😬
  Definitely not.
  That's where:
  👉 Filtering & Sorting become essential.
  🧠 What is Filtering?
  Filtering allows clients to request only the data they need.
  Example:
  GET /products?category=Electronics
  Instead of:
  GET /products
  and receiving everything.
  🧠 What is Sorting?
  Sorting allows clients to control result order.
  Example:
  GET /products?sortBy=price
  or
  GET /products?sortBy=rating
  🔄 Example Flow
  Client Request
  ↓
  Apply Filters
  ↓
  Apply Sorting
  ↓
  Return Results
  💻 Example API Request
  GET /products?
  category=Electronics
  &maxPrice=50000
  &sortBy=rating
  💻 Example Response
  {
  "totalRecords": 120,
  "data": [
  {
  "name": "Laptop",
  "price": 45000,
  "rating": 4.8
  }
  ]
  }
  🎯 Why Filtering Matters
  Without Filtering:
  ❌ Large payloads
  ❌ Slow responses
  ❌ More database work
  With Filtering:
  ✔️ Faster queries
  ✔️ Better user experience
  ✔️ Reduced network traffic
  🎯 Common Sorting Options
  Price Ascending
  Price Descending
  Rating
  Created Date
  Name
  🧠 Common Pitfall
  Allowing unrestricted sorting on every column.
  Example:
  GET /products?sortBy=anyColumn
  This can:
  ❌ Hurt performance
  ❌ Create inefficient queries
  Always whitelist allowed fields.
  🚀 Production Insight
  Most production APIs combine:
  ✔️ Filtering
  ✔️ Sorting
  ✔️ Pagination
  Together.
  Because returning filtered data is good.
  Returning filtered + paginated data is even better.
  💡 Pro Tip
  Keep parameter names simple and consistent.
  Example:
  ?category=
  ?sortBy=
  ?page=
  ?pageSize=
  Predictable APIs are easier to use.
  🎯 Interview Insight
  ❓ Why should filtering happen in the database instead of the application?
  👉 Because databases are optimized to filter large datasets efficiently.
  Fetching everything and filtering in memory wastes resources
  ⚖️ Benefits vs Drawbacks
  Filtering & Sorting
  ✔️ Better performance
  ✔️ Better scalability
  ✔️ Better user experience
  ✔️ Smaller payloads
  No Filtering & Sorting
  ❌ Large responses
  ❌ Slower APIs
  ❌ Poor user experience
  ❌ Increased server load
  🎯 Key Insight
  The best APIs don't just return data.
  They help clients find the right data efficiently.
  🔥 Takeaway:
  👉 Filtering and Sorting improve API usability, performance, and scalability.
  👉 Combined with Pagination, they form the foundation of efficient data retrieval APIs.
  🚀 Day 73:
  👉 Idempotency in APIs — preventing duplicate operations and improving reliability
  [#dotnet](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdotnet&trk=public_post-text) [#aspnetcore](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Faspnetcore&trk=public_post-text) [#webapi](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fwebapi&trk=public_post-text)
  [#backenddevelopment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbackenddevelopment&trk=public_post-text) [#restapi](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Frestapi&trk=public_post-text)
  [#softwareengineering](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsoftwareengineering&trk=public_post-text) [#dotnetdeveloper](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdotnetdeveloper&trk=public_post-text)
  [#100DaysOfCode](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2F100daysofcode&trk=public_post-text) [#LearningInPublic](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearninginpublic&trk=public_post-text)
  [#BackendJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbackendjourney&trk=public_post-text) [#api](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fapi&trk=public_post-text)
  [#SrinijaBuilds](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsrinijabuilds&trk=public_post-text)




  [13](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsrinija-velaga-42312b1b4_dotnet-aspnetcore-webapi-activity-7473794379581816832-Zbm0&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsrinija-velaga-42312b1b4_dotnet-aspnetcore-webapi-activity-7473794379581816832-Zbm0&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsrinija-velaga-42312b1b4_dotnet-aspnetcore-webapi-activity-7473794379581816832-Zbm0&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsrinija-velaga-42312b1b4_dotnet-aspnetcore-webapi-activity-7473794379581816832-Zbm0&trk=public_post_feed-cta-banner-cta)
* [Tran Nam Hung](https://vn.linkedin.com/in/tran-nam-hung-734a2b25b?trk=public_post_feed-actor-name)

  3w

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftran-nam-hung-734a2b25b_a-black-winged-kite-improved-fuzzy-clustering-activity-7478459245236051968-55Zj&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Happy to have published the newest method for clustering the distributional data objects.
  This is the second published article in my series on density functions as objects. It means that the clustering method works on a new distributional space with specific properties relative to traditional statistics. Moreover, these data underscore the need for a new method to address imbalanced data.
  Please feel free to contact me if you have any questions or if we need to discuss further.



  [1](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftran-nam-hung-734a2b25b_a-black-winged-kite-improved-fuzzy-clustering-activity-7478459245236051968-55Zj&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftran-nam-hung-734a2b25b_a-black-winged-kite-improved-fuzzy-clustering-activity-7478459245236051968-55Zj&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftran-nam-hung-734a2b25b_a-black-winged-kite-improved-fuzzy-clustering-activity-7478459245236051968-55Zj&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftran-nam-hung-734a2b25b_a-black-winged-kite-improved-fuzzy-clustering-activity-7478459245236051968-55Zj&trk=public_post_feed-cta-banner-cta)
* [Joachim Schork](https://de.linkedin.com/in/joachim-schork?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjoachim-schork_datavisualization-rprogramminglanguage-activity-7472424295470796800-tlJO&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  In the realm of data analysis, adjusted R-squared stands as a vital tool, helping us navigate the complexity of statistical models. But what exactly is it, and how can it guide our decisions?
  🔍 Understanding Adjusted R-squared:
  - It's a statistical measure that evaluates the goodness of fit of a regression model.
  - Unlike plain R-squared, adjusted R-squared considers the number of predictors in the model, offering a more accurate reflection of model performance.
  ✅ Pros:
  - Takes Complexity into Account: Adjusted R-squared adjusts for the number of predictors, guarding against overfitting.
  - Better Model Comparison: It facilitates fair comparisons between models with different numbers of predictors.
  - Reflects Model Fit: Provides insights into how well the model fits the data, aiding in interpretation.
  ❌ Cons:
  - Can't Detect Overfitting Completely: While it helps mitigate overfitting, it doesn't eradicate the risk entirely.
  - May Penalize Complexity: In some cases, overly penalizing complex models may lead to overly simplistic conclusions.
  🤔 When to Use Adjusted R-squared to Determine Variable Removal:
  - When assessing whether additional variables significantly improve model fit.
  - When aiming to strike a balance between model complexity and explanatory power.
  - When comparing models with different numbers of predictors.
  Consider the graph below, illustrating two different regression models. Based on the adjusted R-squared and model complexity, I would choose the second model, which excludes the 'life' and 'generosity' predictors. This model has a slightly lower adjusted R-squared but maintains a good fit while being less complex, striking a better balance between explanatory power and model simplicity.
  Note: Adjusted R-squared isn't considered state-of-the-art due to the emergence of more advanced statistical techniques and machine learning algorithms that offer greater complexity and flexibility in model evaluation and prediction. However, it remains valuable due to its simplicity, quick insights, and historical context. Its ease of interpretation and ability to aid in fair model comparison make it a practical choice for initial model evaluation and decision-making.
  Explore my webinar titled "Data Analysis & Visualization in R," where I delve into regression model comparison and explain the nuances of adjusted R-squared. Learn more: [https://lnkd.in/eVXD2x78](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FeVXD2x78&urlhash=h426&trk=public_post-text)
  [#datavisualization](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatavisualization&trk=public_post-text) [#rprogramminglanguage](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Frprogramminglanguage&trk=public_post-text) [#datasciencecourse](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatasciencecourse&trk=public_post-text) [#data](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdata&trk=public_post-text)




  [151](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjoachim-schork_datavisualization-rprogramminglanguage-activity-7472424295470796800-tlJO&trk=public_post_social-actions-reactions)







  [28 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjoachim-schork_datavisualization-rprogramminglanguage-activity-7472424295470796800-tlJO&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjoachim-schork_datavisualization-rprogramminglanguage-activity-7472424295470796800-tlJO&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjoachim-schork_datavisualization-rprogramminglanguage-activity-7472424295470796800-tlJO&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjoachim-schork_datavisualization-rprogramminglanguage-activity-7472424295470796800-tlJO&trk=public_post_feed-cta-banner-cta)
* [Madanmohan Tiwari](https://in.linkedin.com/in/madanmohan-tiwari?trk=public_post_feed-actor-name)

  3w

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadanmohan-tiwari_dataanalytics-statistics-learninginpublic-activity-7479874853794181120-TbzX&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  📊 Statistics isn't just about formulas.
  It's about asking better questions from data.
  Day 4 of my Statistics for Data Analysis journey was my biggest session yet. Here's what I learned 👇
  ━━━━━━━━━━━━━━━━━━━━━━
  📌 Z-Score & Area Under the Curve
  The Z-table answers one simple question:
  "What percentage of data lies below a given value?"
  ➡️ Left of Z → Use the Z-table value
  ➡️ Right of Z → 1 − Z-table value
  ➡️ Between two Z-scores → Subtract the smaller value from the larger.
  One table. Endless applications.
  ━━━━━━━━━━━━━━━━━━━━━━
  📌 Central Limit Theorem (CLT)
  One of the most important concepts in statistics.
  No matter how your population data is distributed, if you repeatedly take large enough samples (n ≥ 30) and calculate their means, those means form an approximately Normal Distribution.
  That's why confidence intervals, hypothesis testing, and Z-tests work so well in real-world data.
  CLT connects messy data with reliable statistical inference.
  ━━━━━━━━━━━━━━━━━━━━━━
  📌 Probability Essentials
  Today I covered:
  ✅ Bernoulli Distribution
  ✅ Mutually & Non-Mutually Exclusive Events
  ✅ Independent & Dependent Events
  ✅ Conditional Probability
  The biggest takeaway:
  P(B|A) = P(A∩B) / P(A)
  This formula powers recommendation systems, spam filters, medical diagnosis, and the Naive Bayes algorithm.
  ━━━━━━━━━━━━━━━━━━━━━━
  📌 Permutation vs Combination
  One simple question:
  Does the order matter?
  ✔️ Yes → Permutation
  ✔️ No → Combination
  Simple concept. Huge difference.
  ━━━━━━━━━━━━━━━━━━━━━━
  📌 Covariance & Correlation
  Covariance tells us the direction of a relationship.
  Correlation goes one step further by measuring both direction and strength on a scale of -1 to +1.
  I also learned the difference between:
  📈 Pearson Correlation → Linear, continuous, normally distributed data.
  📊 Spearman Correlation → Rank-based, handles skewed data and outliers better.
  ━━━━━━━━━━━━━━━━━━━━━━
  💡 Biggest takeaway
  The more I learn statistics, the more I realize:
  Statistics isn't background knowledge.
  It is the foundation of Data Science and Machine Learning.
  Every ML model starts with understanding the data first.
  Still learning. Still building. 🚀
  💻 README & Practice → [https://lnkd.in/gtTcuCmM](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FgtTcuCmM&urlhash=bnio&trk=public_post-text)
  If you're also learning Data Analytics or Machine Learning, let's connect! 🤝
  [#DataAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalytics&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#LearningInPublic](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flearninginpublic&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#Probability](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fprobability&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#EDA](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Feda&trk=public_post-text)




  [9](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadanmohan-tiwari_dataanalytics-statistics-learninginpublic-activity-7479874853794181120-TbzX&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadanmohan-tiwari_dataanalytics-statistics-learninginpublic-activity-7479874853794181120-TbzX&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadanmohan-tiwari_dataanalytics-statistics-learninginpublic-activity-7479874853794181120-TbzX&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmadanmohan-tiwari_dataanalytics-statistics-learninginpublic-activity-7479874853794181120-TbzX&trk=public_post_feed-cta-banner-cta)
* [Data Intelligence Factory](https://mx.linkedin.com/company/difactory?trk=public_post_feed-actor-name)

  86 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdifactory_mlmodels-dataengineering-latam-activity-7474581940596310016-aUEA&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  This morning we said the model layer is commodifying and the data layer is the moat — and that the right 2026 question is "is our data ready to be operated on by any model?"
  Here's the 5-check data-layer audit we run before recommending any model.
  Check 1 — Pipeline health. We trace every source feeding the AI use case: end-to-end latency, failure rate over the last 90 days, recovery time. Anything above 0.5% failure or above 24-hour recovery gets flagged red. Most LATAM stacks we audit have 2-3 reds nobody is actively watching.
  Check 2 — Schema drift. We run 90 days of historical schema diffs against the live consumer of the data. Silent column adds, dropped fields, type changes. The number that matters: how many silent breaks happened in the last quarter that nobody caught. The honest answer is usually "more than zero."
  Check 3 — Owner clarity. For each table in the AI use case we ask: who is the named owner accountable for the data quality KPI today? Not "the data team." A specific person. No owner = the data will degrade silently within 90 days of go-live.
  Check 4 — Refresh cadence vs decision cadence. We compare how often the data refreshes against how often the AI is supposed to act on it. If the AI decides hourly and the data refreshes daily, the model layer can't compensate. We flag the mismatch and propose the minimum refresh upgrade.
  Check 5 — Lineage and explainability. For the columns the model will use as features, we trace lineage back to the system of record. If lineage cannot be reconstructed in under 30 minutes, audit/risk will block production. We document the gaps before they become blockers.
  What ships: a 5-page audit report with red/yellow/green per check, a 30-day remediation plan, and an explicit "ready for AI" go/no-go.
  We've run this 14 times in the last 18 months across MX, COL, and CHL. The recommendation has been "fix the data first" 11 of those times. Saved roughly 18 months of model work that would have been thrown away.
  The model is the demo. The data is the moat. The audit is where we'd start.
  Cost: fixed. Timeline: 2 weeks. Outcome: a defensible go/no-go before the model budget is spent.
  → [https://lnkd.in/eVRuVuci](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FeVRuVuci&urlhash=vK8a&trk=public_post-text)
  [#MLModels](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmlmodels&trk=public_post-text) [#DataEngineering](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataengineering&trk=public_post-text) [#LATAM](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flatam&trk=public_post-text)

  + View C2PA information


  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdifactory_mlmodels-dataengineering-latam-activity-7474581940596310016-aUEA&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdifactory_mlmodels-dataengineering-latam-activity-7474581940596310016-aUEA&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdifactory_mlmodels-dataengineering-latam-activity-7474581940596310016-aUEA&trk=public_post_feed-cta-banner-cta)

26,554 followers

[View Profile](https://www.linkedin.com/company/datainterview?trk=public_post_follow-view-profile)
[Connect](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7470142878741102594&trk=public_post_follow)

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
