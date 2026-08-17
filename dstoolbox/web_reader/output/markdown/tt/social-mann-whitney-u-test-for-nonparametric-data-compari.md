---
url: https://www.linkedin.com/posts/what-is-the-mann-whitney-u-test-in-data-share-7468693252179820545-1see/?utm_source=share&utm_medium=member_android&rcm=ACoAACmphIUBo2D4cUQR_ZXSmdR7KMycDh8BUk8
scraped_at: 2026-07-27T14:12:01.051224
depth: 0
---

Mann-Whitney U Test for Nonparametric Data Comparison | DataInterview.com posted on the topic | LinkedIn



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# Mann-Whitney U Test for Nonparametric Data Comparison

This title was summarized by AI from the post below.

[DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_feed-actor-name)

26,554 followers

1mo

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

What is the Mann-Whitney U Test? (in Data interviews)
👋 Let's learn together ↓
The Mann-Whitney U Test is a 𝗻𝗼𝗻𝗽𝗮𝗿𝗮𝗺𝗲𝘁𝗿𝗶𝗰 𝘁𝗲𝘀𝘁 𝘁𝗵𝗮𝘁 𝗰𝗼𝗺𝗽𝗮𝗿𝗲𝘀 𝘁𝘄𝗼 𝗶𝗻𝗱𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝘁 𝗴𝗿𝗼𝘂𝗽𝘀 𝘂𝘀𝗶𝗻𝗴 𝗿𝗮𝗻𝗸𝘀, 𝗻𝗼𝘁 𝗿𝗮𝘄 𝘃𝗮𝗹𝘂𝗲𝘀.
You pool both groups together, rank every observation, then ask: does one group tend to produce larger values than the other? No normality required.
Common misconception: it does NOT test equality of medians. It tests 𝘀𝘁𝗼𝗰𝗵𝗮𝘀𝘁𝗶𝗰 𝗱𝗼𝗺𝗶𝗻𝗮𝗻𝗰𝗲. That is, whether P(X > Y) = P(Y > X). The median interpretation only holds when both distributions share the same shape.
📐 𝗧𝗵𝗲 𝗨 𝗦𝘁𝗮𝘁𝗶𝘀𝘁𝗶𝗰:
U₁ = R₁ - n₁(n₁ + 1) / 2
Use U = min(U₁, U₂) as the test statistic.
Where:
R₁ → sum of ranks for group 1 in the pooled sample
n₁ → number of observations in group 1
U₁ → counts how many (xᵢ, yⱼ) pairs where group 1 beats group 2
U = 0 or U = n₁n₂ signals perfect separation between groups.
For large samples (n₁, n₂ ≥ 8), use the normal approximation:
z = (U - μᵤ) / σᵤ
μᵤ = n₁n₂ / 2
σᵤ = sqrt(n₁n₂(n₁ + n₂ + 1) / 12)
For small samples, use the exact distribution instead.
⚡ 𝗛𝗼𝘄 𝗶𝘁 𝘄𝗼𝗿𝗸𝘀:
① Pool all observations from both groups together
② Rank every value from smallest to largest (ties get averaged ranks)
③ Sum the ranks for group 1 to get R₁
④ Compute U₁ and U₂, take the minimum
⑤ Compare U to a critical value or convert to a z-score
🧐 𝗛𝗼𝘄 𝗶𝘀 𝗶𝘁 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁 𝗳𝗿𝗼𝗺 𝘁𝗵𝗲 𝗶𝗻𝗱𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝘁 𝘀𝗮𝗺𝗽𝗹𝗲𝘀 𝘁-𝘁𝗲𝘀𝘁?
The t-test assumes both groups are normally distributed and compares means directly.
Mann-Whitney makes no normality assumption, works on ordinal data, and compares the full distribution shape, not just the center.
The t-test is more powerful when normality holds. Mann-Whitney is better when data is skewed, has outliers, or comes from Likert scales and response times.
Effect size for Mann-Whitney: report r = z / sqrt(N) where N = n₁ + n₂.
✍️ 𝗪𝗵𝗲𝗻 𝘁𝗼 𝘂𝘀𝗲 𝗠𝗮𝗻𝗻-𝗪𝗵𝗶𝘁𝗻𝗲𝘆:
when your data is ordinal, skewed, or has outliers, when sample sizes are small, or when you can't justify the normality assumption needed for a t-test.
👉 Land Data & AI jobs on [datainterview.com](https://www.linkedin.com/redir/redirect?url=http%3A%2F%2Fdatainterview%2Ecom&urlhash=bCYW&trk=public_post-text)




[71](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_social-actions-reactions)







[1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_social-actions-comments)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

[DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_comment_actor-name)

1mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

👉 Land data, AI, quant jobs on [datainterview.com](http://datainterview.com?trk=public_post_comment-text)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_comment_reply)

1 Reaction

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-mann-whitney-u-test-in-data-activity-7468693253492555776-oQYh&trk=public_post_feed-cta-banner-cta)

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
* [GOKUL DAS](https://in.linkedin.com/in/gokuldas-cmp?trk=public_post_feed-actor-name)

  3w

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgokuldas-cmp_modelevaluation-datascience-mlinterview-activity-7477577246933966848-7Swq&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  [Real Data Interview Question of the Day]
  Role: Data Science / Analytics
  Q: You have two models: Model A has 90% accuracy, Model B has 82% accuracy. Which do you choose?
  The "correct" answer is: it depends.
  But the great answer walks through what it depends on:
  → Class imbalance? If the positive class is 5% of data, 90% accuracy could mean Model A just predicts "no" for everything.
  → Precision vs Recall trade-off? In fraud detection, false negatives (missing fraud) are catastrophic. In spam filtering, false positives (deleting real emails) are worse. Model B might be better for one, A for the other.
  → Business cost of errors? What's the cost of a wrong prediction in each direction?
  → Interpretability? Does the business need to explain its decisions? If so, a simpler Model B might win.
  → Latency and compute? A model that's 8% less accurate but 10x faster in production might be the right call.
  → Calibration? Are the predicted probabilities reliable, not just the class labels?
  Accuracy is a headline. Context is everything.
  Practice questions like these at 👉 [https://lnkd.in/gmbhZhDr](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FgmbhZhDr&urlhash=ZH7I&trk=public_post-text)
  [#ModelEvaluation](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmodelevaluation&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#MLInterview](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmlinterview&trk=public_post-text) [#NextInterview](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fnextinterview&trk=public_post-text)




  [3](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgokuldas-cmp_modelevaluation-datascience-mlinterview-activity-7477577246933966848-7Swq&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgokuldas-cmp_modelevaluation-datascience-mlinterview-activity-7477577246933966848-7Swq&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgokuldas-cmp_modelevaluation-datascience-mlinterview-activity-7477577246933966848-7Swq&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fgokuldas-cmp_modelevaluation-datascience-mlinterview-activity-7477577246933966848-7Swq&trk=public_post_feed-cta-banner-cta)
* [Bernard Muoneme, PhD](https://ng.linkedin.com/in/bernardmuoneme?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fbernardmuoneme_most-data-analysts-dont-fail-interviews-activity-7475050074499358720-8eQH&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Most data analysts don’t fail interviews because of their tools.
  They fail because of one question they freeze on.
  “Walk me through how you’d approach this business problem.”
  Boom, everywhere go first blur.
  This happens to people who can build a dashboard, write a clean SQL query.
  The skills are there. But something is missing
  They jump into the data before they understand the problem.
  They can build the chart but can’t explain why it matters to the business.
  This is the gap nobody talks about.
  A top analyst - ([Mary Komolafe](https://ng.linkedin.com/in/marykomolafe?trk=public_post-text)) is running a 3-day virtual workshop built specifically to close that gap. Not another tools class.
  A workshop on how analysts actually think.
  What it covers:
  • A practical framework for breaking down business problems like an analyst
  • How to ask sharper business and data questions before touching the data
  • The thinking process analysts use to turn numbers into decisions
  • How to use AI to sharpen your analysis without outsourcing your judgment
  • A clearer roadmap for projects and job-ready skills that make you stand out
  It also includes a 30-day implementation track with mentorship and accountability
  Spots: Limited to the first 20 people
  Registration closes July 1st
  Link is in the comments for anyone who wants in.
  If you know someone who has the tools but not the thinking yet, tag them. This might be exactly what they need.
  PS: Virtual or Physical classes? Which do you prefer generally?




  [52](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fbernardmuoneme_most-data-analysts-dont-fail-interviews-activity-7475050074499358720-8eQH&trk=public_post_social-actions-reactions)







  [28 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fbernardmuoneme_most-data-analysts-dont-fail-interviews-activity-7475050074499358720-8eQH&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fbernardmuoneme_most-data-analysts-dont-fail-interviews-activity-7475050074499358720-8eQH&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fbernardmuoneme_most-data-analysts-dont-fail-interviews-activity-7475050074499358720-8eQH&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fbernardmuoneme_most-data-analysts-dont-fail-interviews-activity-7475050074499358720-8eQH&trk=public_post_feed-cta-banner-cta)
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
* [DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_feed-actor-name)

  26,554 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

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




  [133](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-kolmogorov-smirnov-test-in-activity-7470142878741102594-7HHh&trk=public_post_feed-cta-banner-cta)
* [RAHUL GUPTA](https://in.linkedin.com/in/rahul-gupta-343610193?trk=public_post_feed-actor-name)

  1mo

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frahul-gupta-343610193_sdesheetchallenge-dsa-leetcode-activity-7476252576762695680-8p-O&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🗓️ Day 26/45 — [#SDESheetChallenge](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsdesheetchallenge&trk=public_post-text) | Topic: Stack & Queue Part II
  3 problems today. The last two are system design data structures that show up in interviews at scale-focused companies.
  ━━━━━━━━━━━━━━━━━━━━
  ✅ Q8 — Next Smaller Element (Medium)
  Mirror of Next Greater Element — same monotonic stack pattern, different direction.
  • Brute O(N²): nested loop, scan right for first smaller.
  • Optimal O(N): monotonic increasing stack.
  Two approaches I wrote:
  → Left to right with index stack: when arr[i] < arr[stack top], pop and assign arr[i] as the NSE. Intuitive.
  → Right to left with value stack: pop elements ≥ current, stack top = NSE. Classic.
  Both O(N). The pattern generalises: next greater → decreasing stack, next smaller → increasing stack. Traverse direction can be either — just adjust the condition.
  ━━━━━━━━━━━━━━━━━━━━
  ✅ Q9 — LRU Cache (Medium)
  Least Recently Used eviction. get and put in O(1).
  Data structure: HashMap + Doubly Linked List.
  → HashMap: key → node (O(1) access)
  → DLL: maintains access order (most recent at head, least recent at tail)
  On get: move node to front. On put: if key exists, update and move to front. If cache full, delete [tail.prev](http://tail.prev?trk=public_post-text) (LRU), then insert new node at front.
  The dummy head and tail nodes eliminate edge cases for empty list and single node — worth always using in interviews.
  ━━━━━━━━━━━━━━━━━━━━
  ✅ Q10 — LFU Cache (Hard)
  Least Frequently Used eviction — evict the element with lowest access count. Ties broken by LRU.
  Data structure: HashMap + HashMap
  On access: remove node from its frequency list, increment count, insert into frequency+1 list. Track minFreq globally — update it when the minFreq list becomes empty (only possible during insertions, not gets).
  On eviction: remove tail of the minFreq list.
  LFU is harder than LRU because eviction depends on frequency across the entire lifetime, not just recency. The per-frequency DLL handles both the frequency tracking and LRU tie-breaking simultaneously.
  ━━━━━━━━━━━━━━━━━━━━
  LRU and LFU are system design staples. Understanding the implementation — not just the concept — is what separates people in interviews.
  📎 Sheet: [https://lnkd.in/gaQv\_RTv](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FgaQv_RTv&urlhash=fcMP&trk=public_post-text) BY [Raj Vikramaditya](https://in.linkedin.com/in/rajstriver?trk=public_post-text) & [takeUforward](https://in.linkedin.com/company/takeuforward?trk=public_post-text) Team.
  [#DSA](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdsa&trk=public_post-text) [#LeetCode](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fleetcode&trk=public_post-text) [#SDE](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsde&trk=public_post-text) [#Java](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fjava&trk=public_post-text) [#SDESheetChallenge](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsdesheetchallenge&trk=public_post-text) [#TakeUForward](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ftakeuforward&trk=public_post-text) [#LRUCache](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flrucache&trk=public_post-text) [#LFUCache](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Flfucache&trk=public_post-text) [#MonotonicStack](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmonotonicstack&trk=public_post-text) [#SystemDesign](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsystemdesign&trk=public_post-text)




  [2](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frahul-gupta-343610193_sdesheetchallenge-dsa-leetcode-activity-7476252576762695680-8p-O&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frahul-gupta-343610193_sdesheetchallenge-dsa-leetcode-activity-7476252576762695680-8p-O&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frahul-gupta-343610193_sdesheetchallenge-dsa-leetcode-activity-7476252576762695680-8p-O&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Frahul-gupta-343610193_sdesheetchallenge-dsa-leetcode-activity-7476252576762695680-8p-O&trk=public_post_feed-cta-banner-cta)
* [Aman Kumar](https://in.linkedin.com/in/amankumardatainsights?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Famankumardatainsights_dataanalytics-statistics-dataanalyst-activity-7470032769121857536-q33G&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Day 29 | Statistics-9: Three Distributions Every Data Analyst Needs 📊
  Understanding these isn't optional—it's the foundation of credible analysis.
  1️⃣ Chi-Square Test (Independence)
  Tests if two categorical variables are related or random noise.
  Example: Does gender influence product preference?
  Gender | Like | Dislike Male — 40 — 20 Female — 30 — 10
  Process: Calculate expected frequencies → Compare observed vs expected → Get χ² value (0.793). → Compare to critical value (3.841) → Conclusion: No significant relationship.
  For analysts: marketing campaigns, A/B tests, customer segments—you're constantly asking, "Is this real or coincidence?" Chi-square answers it. ✅
  2️⃣ Binomial Distribution
  Probability of X successes in N independent trials.
  Remember: BINS – Binary outcomes, Independent, N fixed trials, Same probability.
  Example: 60% conversion rate. 10 leads contacted. Probability of exactly 6 converting?
  Result: 25.1% chance
  For analysts: Forecasting conversions, predicting churn, quality control. Tells you what's normal vs unusual. ✅
  3️⃣ Poisson Distribution
  Probability of X random events in fixed time/space intervals.
  Example: 4 calls/hour average. Probability of exactly 2 calls next hour?
  Result: 14.64% chance
  For analysts: Capacity planning, anomaly detection, operations. When you expect 4 calls but get 20—Poisson quantifies the abnormality. ✅
  Why This Matters 🎯
  Most analysts use these tests without understanding assumptions or when they fail. Pen-and-paper work exposed what code hides.
  When you truly understand the mechanics:
  You catch violations before running analysis
  You explain results with confidence
  You don't misuse tools and damage credibility
  You know which distribution solves which problem
  This separates hired analysts from passed-over ones. 📈
  Special Thanks to [Ankita Thakkar](https://in.linkedin.com/in/dr-ankita-thakkar-08b3526b?trk=public_post-text)
  [#DataAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalytics&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#DataAnalyst](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalyst&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#FoundationalSkills](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffoundationalskills&trk=public_post-text) [#PythonLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpythonlearning&trk=public_post-text) [#StatisticsBasics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatisticsbasics&trk=public_post-text) [#CareerGrowth](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcareergrowth&trk=public_post-text)



  [5](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Famankumardatainsights_dataanalytics-statistics-dataanalyst-activity-7470032769121857536-q33G&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Famankumardatainsights_dataanalytics-statistics-dataanalyst-activity-7470032769121857536-q33G&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Famankumardatainsights_dataanalytics-statistics-dataanalyst-activity-7470032769121857536-q33G&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Famankumardatainsights_dataanalytics-statistics-dataanalyst-activity-7470032769121857536-q33G&trk=public_post_feed-cta-banner-cta)
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
* [MOHAMUD ABDULLAHI MOHAMED](https://so.linkedin.com/in/mohamud-abdullahi-mohamed-95343063?trk=public_post_feed-actor-name)

  3w

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmohamud-abdullahi-mohamed-95343063_choosing-the-right-statistical-test-the-activity-7477613134342500352-KtNm&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🧠 Choosing the Right Statistical Test
  The infographic titled “Which Statistical Test Should You Use?” by The Data Pulse serves as a comprehensive decision framework for selecting appropriate statistical tests based on data characteristics. It emphasizes starting with the structure of the data not the test name by guiding users through a logical flowchart that begins with identifying the outcome type (continuous or categorical).
  For continuous outcomes, it outlines tests such as the One-Sample t-test, Paired t-test, Independent t-test, Welch’s t-test, ANOVA, and Kruskal–Wallis, depending on group count and data normality.
  For categorical outcomes, it recommends the Chi-Square Test or Fisher’s Exact Test for associations, and Logistic Regression or Multinomial Logistic Regression for predictions. When analyzing relationships, it distinguishes between Pearson Correlation for linear continuous variables and Spearman Correlation for non-linear or ranked data.
  The chart concludes with reminders that test choice is a governance decision, urging analysts to validate assumptions, interpret effect sizes, and consider practical impact underscoring that while tests may be statistically similar, their consequences differ significantly.




  [356](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmohamud-abdullahi-mohamed-95343063_choosing-the-right-statistical-test-the-activity-7477613134342500352-KtNm&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmohamud-abdullahi-mohamed-95343063_choosing-the-right-statistical-test-the-activity-7477613134342500352-KtNm&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmohamud-abdullahi-mohamed-95343063_choosing-the-right-statistical-test-the-activity-7477613134342500352-KtNm&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmohamud-abdullahi-mohamed-95343063_choosing-the-right-statistical-test-the-activity-7477613134342500352-KtNm&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmohamud-abdullahi-mohamed-95343063_choosing-the-right-statistical-test-the-activity-7477613134342500352-KtNm&trk=public_post_feed-cta-banner-cta)
* [Pranav More](https://in.linkedin.com/in/pranav-more-b028092a7?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpranav-more-b028092a7_statistics-datascience-dataanalysis-activity-7472168298487398401-VjPU&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  "The p-value is 0.03 — our result is significant!"
  I've seen this said confidently in presentations — and understood completely wrong. 📊
  The p-value is the most cited and most misunderstood number in data science. Let me fix that.
  ━━━━━━━━━━━━━━━━━━
  What the p-value actually means
  ━━━━━━━━━━━━━━━━━━
  The p-value is the probability of getting your observed result — or something more extreme — IF the null hypothesis were true.
  In plain English:
  p = 0.03 means there's a 3% chance of seeing this result by random chance alone — assuming nothing is actually going on.
  That's it. Nothing more.
  ━━━━━━━━━━━━━━━━━━
  What the p-value does NOT mean
  ━━━━━━━━━━━━━━━━━━
  ❌ It does NOT mean there is a 97% chance your hypothesis is correct
  ❌ It does NOT measure the size or importance of an effect
  ❌ It does NOT prove causation
  ❌ p < 0.05 does NOT mean your result is "real" — it's just a threshold
  These are the exact mistakes made in A/B tests, research papers and product decisions every day.
  ━━━━━━━━━━━━━━━━━━
  Real example — A/B test gone wrong
  ━━━━━━━━━━━━━━━━━━
  Company runs an A/B test on a button colour change.
  Result: p = 0.04 — "statistically significant!" ✅
  They ship the change to all users.
  What they missed:
  → Sample size was only 200 users
  → The actual conversion lift was 0.1% — practically meaningless
  → They ran 30 other tests that week — with 30 tests, getting one false positive at p < 0.05 is almost guaranteed
  The p-value said the result was unlikely by chance.
  It said nothing about whether it mattered.
  ━━━━━━━━━━━━━━━━━━
  The 0.05 threshold — where did it come from?
  ━━━━━━━━━━━━━━━━━━
  Ronald Fisher suggested 0.05 as a convenient threshold in 1925.
  It was never meant to become a universal rule.
  It stuck anyway.
  ━━━━━━━━━━━━━━━━━━
  What to look at instead
  ━━━━━━━━━━━━━━━━━━
  → Effect size — how big is the actual difference?
  → Confidence intervals — what range of values is plausible?
  → Practical significance — does this difference matter in the real world?
  → Sample size — is your test actually powered enough to detect a real effect?
  In Python:
  from scipy import stats
  t\_stat, p\_value = stats.ttest\_ind(group\_a, group\_b)
  # Always pair p-value with effect size — never report it alone
  ━━━━━━━━━━━━━━━━━━
  The rule
  ━━━━━━━━━━━━━━━━━━
  Statistical significance ≠ practical significance.
  A tiny p-value on a meaningless effect is still a meaningless result.
  Have you ever seen a decision made purely because "p < 0.05"? 👇
  [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#DataAnalysis](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalysis&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#ABTesting](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fabtesting&trk=public_post-text) [#HypothesisTesting](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fhypothesistesting&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#OpenToWork](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fopentowork&trk=public_post-text) [#AnalyticsTips](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fanalyticstips&trk=public_post-text) [#DataDriven](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatadriven&trk=public_post-text)




  [3](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpranav-more-b028092a7_statistics-datascience-dataanalysis-activity-7472168298487398401-VjPU&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpranav-more-b028092a7_statistics-datascience-dataanalysis-activity-7472168298487398401-VjPU&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpranav-more-b028092a7_statistics-datascience-dataanalysis-activity-7472168298487398401-VjPU&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpranav-more-b028092a7_statistics-datascience-dataanalysis-activity-7472168298487398401-VjPU&trk=public_post_feed-cta-banner-cta)

26,554 followers

[View Profile](https://www.linkedin.com/company/datainterview?trk=public_post_follow-view-profile)
[Follow](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7468693253492555776&trk=public_post_follow)

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
