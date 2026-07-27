---
url: https://www.linkedin.com/posts/abdullahzahid77_machinelearning-datascience-mlops-share-7465007954086354944-Ikcm/?utm_source=share&utm_medium=member_android&rcm=ACoAACmphIUBo2D4cUQR_ZXSmdR7KMycDh8BUk8
scraped_at: 2026-07-27T14:10:31.622825
depth: 0
---

Is Model Improvement Real or Just Noise? | Abdullah Zahid posted on the topic | LinkedIn



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# Is Model Improvement Real or Just Noise?

This title was summarized by AI from the post below.

[Abdullah Zahid](https://au.linkedin.com/in/abdullahzahid77?trk=public_post_feed-actor-name)

2mo

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

91.2% → 92.1%. Everyone celebrates. New model ships.
But is that improvement actually real, or just noise from a lucky train/test split?
I've been burned by this. Tiny gains can come from random sampling, fold differences, or pure chance, and most teams never bother to check.
Two tests I now run before declaring a winner:
→ Bootstrap testing: resample predictions repeatedly and see if the improvement holds up
→ Wilcoxon signed-rank test: compare models across folds without assuming normality
Some models that looked better on paper didn't survive either test. The gain was real enough to see, but not stable enough to trust.
There's a difference between "the score went up" and "the score meaningfully went up."
In production, only one of those matters.
[#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#MLOps](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmlops&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text)




[50](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_social-actions-reactions)







[1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_social-actions-comments)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

[Andrés Segura-Tinoco](https://co.linkedin.com/in/andres-segura-tinoco/en?trk=public_post_comment_actor-name)

1mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

Great post! This is one of the most common challenges in production ML: choosing between models with very similar performance.
A practical rule I've used is to compare the top two candidates with both bootstrap and Wilcoxon tests, and only declare a winner if both show significance.
Otherwise, I default to the more parsimonious model (fewer features). If the improvement isn't statistically convincing, simplicity is usually the safer bet.
Thanks sharing, [Abdullah Zahid](https://au.linkedin.com/in/abdullahzahid77?trk=public_post_comment-text)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_comment_reply)

1 Reaction

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fabdullahzahid77_machinelearning-datascience-mlops-activity-7465171161472086016-gWAb&trk=public_post_feed-cta-banner-cta)

## More Relevant Posts

* [Aritro Roy](https://www.linkedin.com/in/aritro-roy19?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faritro-roy19_competitiveprogramming-cses-algorithms-activity-7468670513498787844-llRU&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  My big idea to optimize Dijkstra turned out to be from 1969. Here's what 7–8 hours down the rabbit hole taught me.
  Semester wrapped up, so instead of resting I went grinding the CSES problem set. Got through the DP section (at least the ones my brain could handle 😅), moved to graphs, and while revising 0-1 BFS and Dijkstra one question stuck: can I make Dijkstra's priority queue faster?
  I've always been fascinated by heaps; binary, Fibonacci, radix, so chased it. The findings:
  → My "clever" idea (one bucket per distance, kill the log factor) already exists. It's Dial's algorithm. From 1969. Humbling start.
  → The log factor in Dijkstra never actually disappears. You trade it for a dependence on edge weights. Radix heaps do this trade well: for integer weights, Dijkstra becomes effectively linear.
  → I built it in Rust and benchmarked it properly. The benchmark lied to me twice. A debug build flipped which heap "won," and the same binary on two machines gave opposite conclusions. Cache size, not the algorithm, was deciding.
  → The kicker: I rewrote the heap into a version that's asymptotically WORSE on paper, and it ran ~2x faster on a million-node graph. At scale you're not paying for operations, you're paying for cache misses.
  The biggest takeaway wasn't a data structure. It was: measure, then measure again somewhere else, and always write down your build flags.
  Full write-up - code, numbers, and the surprises: [https://lnkd.in/ehv8W7af](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fehv8W7af&urlhash=hQem&trk=public_post-text)
  Code + stress tests + benchmark harness: [https://lnkd.in/eru2WQ7p](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Feru2WQ7p&urlhash=qBYm&trk=public_post-text)
  If you're grinding CSES too, this one started from the graph section. Highly recommend the rabbit holes.
  [#CompetitiveProgramming](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcompetitiveprogramming&trk=public_post-text) [#CSES](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcses&trk=public_post-text) [#Algorithms](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Falgorithms&trk=public_post-text) [#Rust](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Frust&trk=public_post-text) [#DataStructures](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatastructures&trk=public_post-text) [#Dijkstra](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdijkstra&trk=public_post-text)

  + View C2PA information


  [46](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faritro-roy19_competitiveprogramming-cses-algorithms-activity-7468670513498787844-llRU&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faritro-roy19_competitiveprogramming-cses-algorithms-activity-7468670513498787844-llRU&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faritro-roy19_competitiveprogramming-cses-algorithms-activity-7468670513498787844-llRU&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Faritro-roy19_competitiveprogramming-cses-algorithms-activity-7468670513498787844-llRU&trk=public_post_feed-cta-banner-cta)
* [Jason Kwiatkowski](https://www.linkedin.com/in/jason-kwiatkowski-309265307?trk=public_post_feed-actor-name)

  1mo

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjason-kwiatkowski-309265307_slideshow-for-project-activity-7467378827665543168-mGc0&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Recent group project with [Nathaniel Cruz](https://www.linkedin.com/in/nathaniel-cruz-924516346?trk=public_post-text)
  We coded a drop-detection model for the flipper zero.
  While that itself isn’t hard and it’s just the application of the core foundations. We had to build each part from scratch to make it happen; the data collector, the model (trainer, tuner, tester, etc), the inference of the model.
  No existing ml libraries, just the foundations of CNN.
  The slide show is below and so is the repo.
  [https://lnkd.in/gw9gCknE](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fgw9gCknE&urlhash=x5y_&trk=public_post-text)




  [1](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjason-kwiatkowski-309265307_slideshow-for-project-activity-7467378827665543168-mGc0&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjason-kwiatkowski-309265307_slideshow-for-project-activity-7467378827665543168-mGc0&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjason-kwiatkowski-309265307_slideshow-for-project-activity-7467378827665543168-mGc0&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjason-kwiatkowski-309265307_slideshow-for-project-activity-7467378827665543168-mGc0&trk=public_post_feed-cta-banner-cta)
* [MD Zeeshan](https://in.linkedin.com/in/md-zeeshan-1431262a5?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmd-zeeshan-1431262a5_leetcode-cpp-datastructures-activity-7471644148467675136-t2sM&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  The visual definition of "Brute Force vs. Optimal"
  If you ever wonder why Data Structures and Algorithms matter, just look at this submission history. Getting an "Accepted" status is great, but it's rarely the finish line.
  Here is the breakdown of the journey:
  🔴 The Struggle: Runtime Errors & Wrong Answers. (We've all been there).
  🟡 Brute Force (1676 ms | 508.1 MB): The "just make it work" phase. It checks every possibility. It’s resource-heavy, but it gets the green text.
  🟢 Optimal (16 ms | 17.6 MB): The "make it fast" phase. By dropping redundant calculations, optimizing space, and picking the right data structure, the runtime dropped by 99%!
  The takeaway: Brute force solves the problem.
  Better approaches refine the logic.
  Optimal solutions scale.
  Never settle for the first "Accepted". Keep pushing the code! 💻💡
  [#LeetCode](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fleetcode&trk=public_post-text) [#CPP](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcpp&trk=public_post-text) [#DataStructures](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatastructures&trk=public_post-text) [#Algorithms](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Falgorithms&trk=public_post-text) [#ProblemSolving](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fproblemsolving&trk=public_post-text) [#SoftwareDevelopment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsoftwaredevelopment&trk=public_post-text) [#CodingJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcodingjourney&trk=public_post-text)




  [71](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmd-zeeshan-1431262a5_leetcode-cpp-datastructures-activity-7471644148467675136-t2sM&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmd-zeeshan-1431262a5_leetcode-cpp-datastructures-activity-7471644148467675136-t2sM&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmd-zeeshan-1431262a5_leetcode-cpp-datastructures-activity-7471644148467675136-t2sM&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmd-zeeshan-1431262a5_leetcode-cpp-datastructures-activity-7471644148467675136-t2sM&trk=public_post_feed-cta-banner-cta)
* [Shailesh Kumar](https://in.linkedin.com/in/contactshailesh-kumar?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcontactshailesh-kumar_datastructures-algorithms-codingjourney-activity-7472323007361900545-EkGm&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Today, I was tackling the "K-Sized Subarray Maximum" problem.
  My initial approach was intuitive: use a while loop to slide through the array and find the maximum in every window of size K. It worked perfectly for smaller test cases! But when I hit submit...
  💥 Time Limit Exceeded (TLE).
  isn't a failure—it's an invitation to optimize. It means your logic is correct, but your algorithm is too slow to handle large-scale data within the required time limit (usually 1 second).
  [#DataStructures](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatastructures&trk=public_post-text) [#Algorithms](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Falgorithms&trk=public_post-text) [#CodingJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcodingjourney&trk=public_post-text) [#ProblemSolving](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fproblemsolving&trk=public_post-text) [#SoftwareEngineering](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsoftwareengineering&trk=public_post-text) [#ContinuousLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcontinuouslearning&trk=public_post-text)




  [4](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcontactshailesh-kumar_datastructures-algorithms-codingjourney-activity-7472323007361900545-EkGm&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcontactshailesh-kumar_datastructures-algorithms-codingjourney-activity-7472323007361900545-EkGm&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcontactshailesh-kumar_datastructures-algorithms-codingjourney-activity-7472323007361900545-EkGm&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcontactshailesh-kumar_datastructures-algorithms-codingjourney-activity-7472323007361900545-EkGm&trk=public_post_feed-cta-banner-cta)
* [Joshua Egan](https://www.linkedin.com/in/josheganai?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjosheganai_consensus-is-expensive-when-you-have-three-activity-7466665180568072193-Tlqy&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Consensus is expensive when you have three independent models and they disagree on every signal.
  That was the miscalculation — I built the multi-model gate thinking disagreement was noise, a sign to filter harder. Instead it became the bottleneck. Two models (Claude for fundamentals, a smaller local inference engine for speed) would flag a trade setup, the third (Anthropic's vision model for chart analysis) would reject it or stay silent. The gate required unanimous sign-off. Reasonable on paper. Catastrophic in practice.
  What actually happened: signals were queuing. Not because the models were slow — they run async on separate containers — but because I'd starved the reconciliation loop. The arbiter (a tiny Python coroutine that waits for all three verdicts before clearing a signal to dispatch) had no timeout, no fallback rank order, no "two-out-of-three is enough" rule. Just infinite wait on the third model. Some nights the vision model would lag 8–12 seconds on chart fetch (Hetzner storage gateway hiccup, turns out), and the queue would pile up. Signals aged out. The trading window closed. Nothing moved.
  The fix wasn't elegant. Added a hard 2-second ceiling on the arbiter, rank-weighted the models (fundamentals 40%, speed 35%, vision 25%), and let majority consensus clear the gate. Dropped model count from three to a weighted trio. Lost the "all agree" guarantee. Gained dispatch latency back — now sub-200ms even under storage lag.
  The dumb part is that I'd built circuit breakers and health monitors for agent crashes, but not for silent queueing. Consensus is infrastructure too. Consensus needs a timer.
  [https://lnkd.in/d5f5A4eb](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fd5f5A4eb&urlhash=Zrbh&trk=public_post-text)



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjosheganai_consensus-is-expensive-when-you-have-three-activity-7466665180568072193-Tlqy&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjosheganai_consensus-is-expensive-when-you-have-three-activity-7466665180568072193-Tlqy&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fjosheganai_consensus-is-expensive-when-you-have-three-activity-7466665180568072193-Tlqy&trk=public_post_feed-cta-banner-cta)
* [Hugo Rodriguez Donato](https://es.linkedin.com/in/hugorodriguezdonato?trk=public_post_feed-actor-name)

  1mo

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fhugorodriguezdonato_github-hugord7churn-project-end-to-end-activity-7474071868556210176-ma2k&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  I’ve just finished an end-to-end machine learning project on bank customer churn prediction and it taught me much more than just how to train a model.
  Here’s what I worked on:
  - Data understanding
  - Exploratory data analysis
  - Preprocessing
  - Model training and evaluation
  - Business interpretation of the final model
  I tested several models:
  Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
  After comparing them with metrics such as precision, recall, F1-score, and ROC-AUC, I selected Random Forest as the best model because it gave the best balance between detecting churners and maintaining solid overall performance.
  Some of the most important insights I found were:
  - Age was the strongest predictor of churn
  - Number of products had a major influence on customer retention
  - Inactive members were much more likely to leave
  - Balance and geography, especially customers from Germany, also played an important role
  What I learned most from this project:
  - Accuracy alone is not enough when the target is imbalanced
  - Preprocessing decisions have a big impact on model quality
  - Model selection is about trade-offs, not just picking the highest score
  - Interpreting the model in business terms is just as important as building it
  This project helped me strengthen my skills in pandas, EDA, preprocessing, classification models, model evaluation.
  I’ve also published the full project on my GitHub account: [https://lnkd.in/exGx9zhP](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FexGx9zhP&urlhash=qNeR&trk=public_post-text) where you can find the notebooks, workflow, and results from start to finish.
  It also reminded me that a good ML project is not only about building a model, it’s about understanding the problem, explaining the results, and connecting them to real decisions.
  Have you worked on a churn prediction project before? What metric would you prioritize most in this kind of problem: recall, precision, or F1-score?
  [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#ScikitLearn](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fscikitlearn&trk=public_post-text) [#EDA](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Feda&trk=public_post-text) [#ChurnPrediction](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fchurnprediction&trk=public_post-text) [#PortfolioProject](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fportfolioproject&trk=public_post-text)



  [7](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fhugorodriguezdonato_github-hugord7churn-project-end-to-end-activity-7474071868556210176-ma2k&trk=public_post_social-actions-reactions)







  [1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fhugorodriguezdonato_github-hugord7churn-project-end-to-end-activity-7474071868556210176-ma2k&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fhugorodriguezdonato_github-hugord7churn-project-end-to-end-activity-7474071868556210176-ma2k&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fhugorodriguezdonato_github-hugord7churn-project-end-to-end-activity-7474071868556210176-ma2k&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fhugorodriguezdonato_github-hugord7churn-project-end-to-end-activity-7474071868556210176-ma2k&trk=public_post_feed-cta-banner-cta)
* [Alishba Riasat](https://pk.linkedin.com/in/alishba-riasat?trk=public_post_feed-actor-name)

  1mo

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Falishba-riasat_datascience-machinelearning-formula1-activity-7472873726255063040-nAy3&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  I was supposed to build a simple data science project.
  It didn’t stay simple.
  It evolved into a Formula 1 race intelligence system that turns raw telemetry into actionable strategy insights
  Introducing Apex 1 Analytics
  What it does:
  • Ingests real Formula 1 race + qualifying data using FastF1
  • Processes driver laps, weather conditions, and race results
  • Analyzes performance using speed, RPM, throttle, and braking telemetry
  • Models tyre degradation using lap time vs tyre life correlation
  • Detects performance “cliffs” in race pace
  • Tracks position changes from grid to finish
  • Builds race trace models for consistency + pace evolution
  Machine learning layer:
  • K-Means clustering on telemetry data (braking zones, corners, straights)
  • Random Forest model for pit-stop probability prediction
  • Hybrid ML + physics-based tyre degradation modeling
  • Strategy signals derived from tyre age and lap behavior
  Engineering layer:
  • Spatial track mapping using X/Y telemetry coordinates
  • Lap-by-lap driver performance comparison
  • Race trace modeling using time-gap evolution
  • Full telemetry profiling (speed, RPM, throttle, brake)
  Stack:
  Frontend: HTML, CSS, JavaScript ([Chart.js](http://Chart.js?trk=public_post-text))
  Backend: FastAPI + Python (Jinja2 templates)
  ML: Scikit-learn (KMeans, RandomForest)
  Data: FastF1, Pandas, NumPy
  A group project that I took on and rebuilt — with [Noor ul Ain](https://pk.linkedin.com/in/noor-ul-ain-431b05317?trk=public_post-text) & [Hafsa Tayyab](https://pk.linkedin.com/in/hafsa-tayyab-4ab566317?trk=public_post-text)
  [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#Formula1](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fformula1&trk=public_post-text) [#FastF1](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffastf1&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text)

  …more

  ![](https://media.licdn.com/dms/image/v2/D4D05AQGstcZm1tOryA/videocover-low/B4DZ7T6_iWGYBI-/0/1781671922972?e=2147483647&v=beta&t=-8GNFT1saITGFx-IdJM3CMiBIKpP80NOROOmrw1K1w0)Play Video

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



  [35](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Falishba-riasat_datascience-machinelearning-formula1-activity-7472873726255063040-nAy3&trk=public_post_social-actions-reactions)







  [13 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Falishba-riasat_datascience-machinelearning-formula1-activity-7472873726255063040-nAy3&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Falishba-riasat_datascience-machinelearning-formula1-activity-7472873726255063040-nAy3&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Falishba-riasat_datascience-machinelearning-formula1-activity-7472873726255063040-nAy3&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Falishba-riasat_datascience-machinelearning-formula1-activity-7472873726255063040-nAy3&trk=public_post_feed-cta-banner-cta)
* [Arches AI](https://www.linkedin.com/company/arches-ai?trk=public_post_feed-actor-name)

  206 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Farches-ai_most-rag-debugging-still-starts-too-late-activity-7470963080366219266-uEfz&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Most RAG debugging still starts too late. Teams see a wrong answer, swap models, adjust the prompt, and only then inspect whether the right source text was ever retrieved.
  Arches usually separates the failure into three buckets before touching generation: the answer is missing from the index, the answer is indexed but not retrieved, or the answer was retrieved and then misused. That split changes the fix. Missing tables need better parsing. Missed exact identifiers need hybrid search. Ignored context needs prompt and ranking work. Treating all three as “the model hallucinated” burns time in the wrong layer.
  A practical first pass is 50 to 100 failed questions, reviewed by hand. For each one, record whether the source exists, whether it appeared in top-k, where the reranker placed it, and what the model saw. The takeaway is that retrieval systems should be debugged with source-level evidence, not answer vibes.



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Farches-ai_most-rag-debugging-still-starts-too-late-activity-7470963080366219266-uEfz&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Farches-ai_most-rag-debugging-still-starts-too-late-activity-7470963080366219266-uEfz&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Farches-ai_most-rag-debugging-still-starts-too-late-activity-7470963080366219266-uEfz&trk=public_post_feed-cta-banner-cta)
* [Kaushik Kumar Venkatesan Premkumar](https://uk.linkedin.com/in/kaushikkumarvp?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fkaushikkumarvp_machinelearning-datascience-fastapi-activity-7465341622982213633-Wg-E&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🚀 Just shipped my first end-to-end machine learning project: a credit
  card fraud detector.
  The dataset has 284,807 transactions. Only 492 are fraud — 0.17%, or
  roughly 1 in 578.
  A model that predicts "not fraud" for everything is 99.83% accurate.
  And completely useless.
  That single fact shaped every decision in the project.
  Here's what I built end-to-end:
  🔹 Stratified data pipeline with no-leakage preprocessing
  🔹 Logistic regression baseline (PR-AUC 0.683)
  🔹 XGBoost contender (PR-AUC 0.877 on held-out test set — +22%)
  🔹 Empirically benchmarked SMOTE vs undersampling vs scale\_pos\_weight
  (SMOTE actually hurt; "do nothing" almost won)
  🔹 Cost-minimizing threshold tuning (FN=$100, FP=$10) — picked t=0.035,
  not the default 0.5
  🔹 FastAPI service with Pydantic validation, 15-test pytest suite
  🔹 Dockerized + deployed on Render with auto-deploy
  🔹 [Next.js](http://Next.js?trk=public_post-text) + TypeScript + Tailwind frontend on Vercel
  🔗 Demo + code — link in the first comment
  The biggest lesson? Most tutorials stop at the trained model.
  The interesting engineering — picking the right metric, choosing a
  threshold from a cost model, structuring a clean three-layer
  architecture, writing tests that catch real bugs — happens \*after\*
  the model works.
  Open to feedback. What did I miss?
  [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#FastAPI](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffastapi&trk=public_post-text) [#XGBoost](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fxgboost&trk=public_post-text) [#FraudDetection](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Ffrauddetection&trk=public_post-text)




  [8](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fkaushikkumarvp_machinelearning-datascience-fastapi-activity-7465341622982213633-Wg-E&trk=public_post_social-actions-reactions)







  [3 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fkaushikkumarvp_machinelearning-datascience-fastapi-activity-7465341622982213633-Wg-E&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fkaushikkumarvp_machinelearning-datascience-fastapi-activity-7465341622982213633-Wg-E&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fkaushikkumarvp_machinelearning-datascience-fastapi-activity-7465341622982213633-Wg-E&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fkaushikkumarvp_machinelearning-datascience-fastapi-activity-7465341622982213633-Wg-E&trk=public_post_feed-cta-banner-cta)
* [Pawel Bulowski](https://pl.linkedin.com/in/pawel-bulowski-ai-agents?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpawel-bulowski-ai-agents_%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%259F%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%259E-%25F0%259D%2590%25AD%25F0%259D%2590%25AE%25F0%259D%2590%25A7%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%25A7%25F0%259D%2590%259E%25F0%259D%2590%25B0-%25F0%259D%2590%25AD%25F0%259D%2590%25A8%25F0%259D%2590%25A8%25F0%259D%2590%25A5%25F0%259D%2590%25AC-activity-7472755569767018496-m8bb&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  𝐍𝐨 𝐟𝐢𝐧𝐞-𝐭𝐮𝐧𝐢𝐧𝐠. 𝐍𝐨 𝐧𝐞𝐰 𝐭𝐨𝐨𝐥𝐬. 𝐉𝐮𝐬𝐭 𝐚 𝐛𝐞𝐭𝐭𝐞𝐫 𝐰𝐚𝐲 𝐭𝐨 𝐜𝐚𝐥𝐥 𝐭𝐡𝐞 𝐨𝐧𝐞𝐬 𝐲𝐨𝐮 𝐚𝐥𝐫𝐞𝐚𝐝𝐲 𝐡𝐚𝐯𝐞. 🤔
  NVIDIA's new 𝐒𝐩𝐚𝐭𝐢𝐚𝐥𝐂𝐥𝐚𝐰 makes a sharp point: for spatial reasoning, the bottleneck isn't model size or tool count — it's the 𝐚𝐜𝐭𝐢𝐨𝐧 𝐢𝐧𝐭𝐞𝐫𝐟𝐚𝐜𝐞 through which a VLM uses its tools.
  It's training-free. The agent writes one Python cell per step into a persistent kernel, inspects the intermediate masks, depth maps and plots, then revises before answering — instead of committing to one full program up front or filling in rigid JSON tool calls.
  𝐓𝐡𝐞 𝐧𝐮𝐦𝐛𝐞𝐫𝐬:
  ➡️ 59.9% avg across 20 spatial benchmarks — +11.2 pts over the prior best spatial agent, same backbone
  ➡️ Hold the toolset fixed, swap only the interface: code-as-action 59.9% > structured tool-calls 56.7% > single-pass code 55.2% > no tools 53.4%
  ➡️ Gains hold across 6 VLM backbones from 26B to 397B params — zero per-model tuning
  ➡️ Biggest lifts exactly where you'd want them: dynamic 4D video (DSI-Bench +18.3pp) and multi-view reasoning (MindCube +14.3pp)
  Before reaching for a bigger model or a longer tool list, look at 𝐡𝐨𝐰 your agent is allowed to act. Compose, inspect, revise beats raw scale.
  [https://lnkd.in/d5DPThmR](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fd5DPThmR&urlhash=daxd&trk=public_post-text)




  [24](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpawel-bulowski-ai-agents_%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%259F%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%259E-%25F0%259D%2590%25AD%25F0%259D%2590%25AE%25F0%259D%2590%25A7%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%25A7%25F0%259D%2590%259E%25F0%259D%2590%25B0-%25F0%259D%2590%25AD%25F0%259D%2590%25A8%25F0%259D%2590%25A8%25F0%259D%2590%25A5%25F0%259D%2590%25AC-activity-7472755569767018496-m8bb&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpawel-bulowski-ai-agents_%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%259F%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%259E-%25F0%259D%2590%25AD%25F0%259D%2590%25AE%25F0%259D%2590%25A7%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%25A7%25F0%259D%2590%259E%25F0%259D%2590%25B0-%25F0%259D%2590%25AD%25F0%259D%2590%25A8%25F0%259D%2590%25A8%25F0%259D%2590%25A5%25F0%259D%2590%25AC-activity-7472755569767018496-m8bb&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpawel-bulowski-ai-agents_%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%259F%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%259E-%25F0%259D%2590%25AD%25F0%259D%2590%25AE%25F0%259D%2590%25A7%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%25A7%25F0%259D%2590%259E%25F0%259D%2590%25B0-%25F0%259D%2590%25AD%25F0%259D%2590%25A8%25F0%259D%2590%25A8%25F0%259D%2590%25A5%25F0%259D%2590%25AC-activity-7472755569767018496-m8bb&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fpawel-bulowski-ai-agents_%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%259F%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%259E-%25F0%259D%2590%25AD%25F0%259D%2590%25AE%25F0%259D%2590%25A7%25F0%259D%2590%25A2%25F0%259D%2590%25A7%25F0%259D%2590%25A0-%25F0%259D%2590%258D%25F0%259D%2590%25A8-%25F0%259D%2590%25A7%25F0%259D%2590%259E%25F0%259D%2590%25B0-%25F0%259D%2590%25AD%25F0%259D%2590%25A8%25F0%259D%2590%25A8%25F0%259D%2590%25A5%25F0%259D%2590%25AC-activity-7472755569767018496-m8bb&trk=public_post_feed-cta-banner-cta)

1,677 followers

* [33 Posts](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fin%2Fabdullahzahid77%2Frecent-activity%2F&trk=public_post_follow-posts)

[View Profile](https://au.linkedin.com/in/abdullahzahid77?trk=public_post_follow-view-profile)
[Connect](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7465171161472086016&trk=public_post_follow)

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
