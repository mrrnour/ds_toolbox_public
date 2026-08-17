---
url: https://www.linkedin.com/posts/adrianolszewski_dear-aspiring-data-scientists-when-you-perform-share-7466889446831980545-lQMb/?utm_source=share&utm_medium=member_android&rcm=ACoAACmphIUBo2D4cUQR_ZXSmdR7KMycDh8BUk8
scraped_at: 2026-07-27T14:10:43.511944
depth: 0
---

Dear Aspiring Data Scientists, when you perform ANOVA/ANCOVA and it turns out that your per-group variances are heterogeneous (non-equal), you really DO NOT HAVE TO switch to non-parametric, e.g… | Adrian Olszewski



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# Adrian Olszewski’s Post

[Adrian Olszewski](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_feed-actor-name)

1mo

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

Dear Aspiring Data Scientists, when you perform ANOVA/ANCOVA and it turns out that your per-group variances are heterogeneous (non-equal), you really DO NOT HAVE TO switch to non-parametric, e.g. rank-based methods.
⚠️ Rank methods answer a different question - about stochastic superiority, not means or medians (unless compared groups are IID, only shifted by location).
⚠️If your intention was to test "difference in locations", rank methods are ALSO sensitive to unequal variances (causing superiority), and can reject H0 for equal means or medians.
⚠️Have typically lower power for Gaussian data (magnitudes ignored).
⚠️ Struggle with tied observations.
Solution?
Try replacing OLS with GLS (Generalized Least Square) estimation for such scenario or OLS with Heteroscedasticity-consistent (sandwich) standard errors. With just a few more lines of R code you can fit a GLS model and use it to obtain both pairwise comparisons and ANOVA (joint effect, main effect).
On my GitHub I showed a trivial example, comparing the output of a pairwise Welch t test with the GLS-fit model (with Satterthwaite degrees of freedom), and the ANOVAs: [https://lnkd.in/dtFg2-Cs](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FdtFg2-Cs&urlhash=b07d&trk=public_post-text)
See, how easy is that?
Note, however:
1) GLS incorporates the covariance estimator into the estimation of beta coefficients, keeping them BLUE as long as Gauss-Markov assumptions with correctly specified covariance matrix is correct. OLS + HCx doesn't do that.
2) With GLS you can specify precisely the covariance structure (not only for heteroscedasticity but also for correlations) or choose "unstructured" pattern. In our case GLS fits just naturally, but when you fail - it loses efficiency.
HC is good if you don't know/anticipate any concrete pattern or don't want to bother with.
3) HC is asymptotic, so you'll need enough data. "Enough" may mean 30 or 3000 - hard to say. There exist a few small-sample corretions, like HC1-3, Mancl-de Rouen (often used in the analysis of longitudinal clinical trials).
4) Heteroskedasticity means the Gaussian likelihood is misspecified. As a consequence, anything based on log-likelihood may be affected. Likelihood ratio tests may no longer be guaranteed to follow chi2 distribution. Briefly, HC estimators fix the Wald's inference, but NOT the LRT.
5) Under missing data GLS/MMRM can remain valid under MAR when correctly specified. HC is valid after an additional strategy such as weighting, imputation, or MCAR (complete-case) assumptions.
Remember:
➡️ OLS + HC: You trust the mean model and don’t model variance, fixing inference externally. HC is agnostic about variance structure.
➡️ GLS: You model variance explicitly and check whether that model explains the residual structure. GLS is "committed" to a variance structure.
---
PS: I'm sorry for a few nonsenses in the AI-generated pic, but I noticed that colourful posters catch attention, correct or not 😁




[132](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_social-actions-reactions)







[8 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_social-actions-comments)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

[Saeid Shahraz](https://www.linkedin.com/in/saeid-shahraz-3b819624?trk=public_post_comment_actor-name)

1mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

Great post!
From a regulatory perspective (ICH E9/E9(R1)), how do you see the choice between GLS and OLS+HC affecting estimand discussions? Does explicitly modeling variance structure through GLS provide better alignment with intercurrent event handling strategies?

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_reply)

1 Reaction

[Aziz Ali](https://se.linkedin.com/in/aziz-ali-271b28113?trk=public_post_comment_actor-name)

1mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

Thanks for highlighting this and offering tips. Among econometricians, this is standard, but in the biostatistics community, switching to a rank-based or other non-parametric approach is the common practice.

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_reply)

1 Reaction

[Shrikant Deshmukh](https://in.linkedin.com/in/shrikant-deshmukh-b9a591352?trk=public_post_comment_actor-name)

1mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

Thanks for sharing, one question weltch t-test and pairwise t-tets returns the same results. No multiplicity adjustment?

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_comment_reply)

1 Reaction

[See more comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_see-more-comments)

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fadrianolszewski_dear-aspiring-data-scientists-when-you-perform-activity-7466889448044158976-pxam&trk=public_post_feed-cta-banner-cta)

## More Relevant Posts

* [Michael Frank](https://www.linkedin.com/in/michael-frank-0b634b73?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmichael-frank-0b634b73_openscience-researchmethods-psychology-activity-7474852044642611201-3BCF&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  If p < 0.05, you "reject the null."
  That's about all most stats courses teach you to do with a result.
  This is week 6 of a weekly Experimentology series I'm running through the spring and summer, sharing one chapter at a time. Today is Ch 6, on statistical inference — with Maya Mathur as cowriter.
  The chapter argues that null hypothesis significance testing (NHST), while ubiquitous, is the wrong default for most of what we want to do with experiments. It gives you exactly one bit of information: did you reject the null or not? That's a lot of effort for one bit. Considering effect sizes, their precision, and visualizations gives a much richer sense of what actually happened in your data.
  We don't argue for tossing out NHST entirely. It has its uses — particularly when the question really is binary (does this intervention have any effect?). But many of the field's analytic pathologies trace back to using NHST when the underlying question is actually about magnitude or precision.
  We walk through alternatives. Bayes Factors quantify the relative support for two hypotheses given the data — a BF of 3 means three times as much evidence for H1 over H0. They're continuous, comparative, and don't need a binary cutoff. Confidence intervals (and Bayesian credible intervals) shift the question from "is it significant?" to "how big is the effect, with what precision?"
  The chapter also walks through common p-value misinterpretations. Two of the most common: "p = 0.05 means the null has a 5% chance of being true" (no — that's a posterior, not a likelihood), and "p > 0.05 lets us accept the null" (also no — failing to reject ≠ evidence for the null). Goodman has a wonderful "dirty dozen" list of these.
  The chapter's accident report is Daryl Bem's 2011 precognition paper, which reported nine experiments showing that people could anticipate stimuli before they appeared. Wagenmakers and colleagues showed that the paper combined analytic flexibility, weak Bayes Factors, and (for any reasonable prior) an extraordinarily low prior probability. Together those undercut the conclusion. Bem's paper helped kick off what we now call the replication crisis.
  The framing we end on: most of the time, the right question isn't "is this effect significant?" but "how big is the effect, and how confident are we?" That estimation framing connects back to Ch 5 and forward to Ch 7 on models.
  📖 Read Ch 6: [https://lnkd.in/gcGvZixY](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FgcGvZixY&urlhash=osvl&trk=public_post-text)
  [#OpenScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fopenscience&trk=public_post-text) [#ResearchMethods](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fresearchmethods&trk=public_post-text) [#Psychology](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpsychology&trk=public_post-text) [#Statistics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstatistics&trk=public_post-text) [#HigherEducation](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fhighereducation&trk=public_post-text)




  [128](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmichael-frank-0b634b73_openscience-researchmethods-psychology-activity-7474852044642611201-3BCF&trk=public_post_social-actions-reactions)







  [5 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmichael-frank-0b634b73_openscience-researchmethods-psychology-activity-7474852044642611201-3BCF&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmichael-frank-0b634b73_openscience-researchmethods-psychology-activity-7474852044642611201-3BCF&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmichael-frank-0b634b73_openscience-researchmethods-psychology-activity-7474852044642611201-3BCF&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmichael-frank-0b634b73_openscience-researchmethods-psychology-activity-7474852044642611201-3BCF&trk=public_post_feed-cta-banner-cta)
* [Milvus, created by Zilliz](https://www.linkedin.com/company/the-milvus-project?trk=public_post_feed-actor-name)

  14,352 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthe-milvus-project_with-the-same-multi-vector-model-and-the-activity-7468323244329050112-Uuyi&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  With the same multi-vector model, and the same dataset, nDCG@10 can drop from 0.701 to 0.109 — roughly a 6x gap. Why? It's because you changed the approximate retrieval strategy.
   𝗜𝗻 𝗺𝘂𝗹𝘁𝗶-𝘃𝗲𝗰𝘁𝗼𝗿 𝗿𝗲𝘁𝗿𝗶𝗲𝘃𝗮𝗹, 𝗽𝗶𝗰𝗸𝗶𝗻𝗴 𝘁𝗵𝗲 𝘄𝗿𝗼𝗻𝗴 𝘀𝘁𝗿𝗮𝘁𝗲𝗴𝘆 𝗰𝗮𝗻 𝗰𝗼𝘀𝘁 𝘆𝗼𝘂 𝗺𝗼𝗿𝗲 𝘁𝗵𝗮𝗻 𝗽𝗶𝗰𝗸𝗶𝗻𝗴 𝘁𝗵𝗲 𝘄𝗿𝗼𝗻𝗴 𝗺𝗼𝗱𝗲𝗹.
  Multi-vector models like ColBERT turn every token in a document into its own vector. You can't just load millions of token vectors into an ANN index and search, because the score that matters is document-level MaxSim — for each query token, find its closest token in the document, then sum.
  So every approach runs in two stages: an approximate search picks candidate documents first, then exact MaxSim re-ranks them. The strategies only differ in how they do that first step.
  𝗪𝗲 𝘁𝗲𝘀𝘁𝗲𝗱 𝘁𝗵𝗿𝗲𝗲: 𝗧𝗼𝗸𝗲𝗻𝗔𝗡𝗡 (𝗶𝗻𝗱𝗲𝘅 𝗲𝘃𝗲𝗿𝘆 𝘁𝗼𝗸𝗲𝗻 𝘃𝗲𝗰𝘁𝗼𝗿 𝗱𝗶𝗿𝗲𝗰𝘁𝗹𝘆), 𝗠𝗨𝗩𝗘𝗥𝗔 (𝗿𝗮𝗻𝗱𝗼𝗺-𝗽𝗿𝗼𝗷𝗲𝗰𝘁𝗶𝗼𝗻 𝗰𝗼𝗺𝗽𝗿𝗲𝘀𝘀𝗶𝗼𝗻), 𝗟𝗘𝗠𝗨𝗥 (𝘁𝗿𝗮𝗶𝗻 𝗮𝗻 𝗠𝗟𝗣 𝘁𝗼 𝗰𝗼𝗺𝗽𝗿𝗲𝘀𝘀).
  On LoTTE, with Jina-ColBERT-v2 held fixed:
  📈 TokenANN — nDCG@10 = 0.701
  📉 LEMUR — nDCG@10 = 0.109
  The only thing that changed is the first-stage approximation. For scale: moving from a plain dense model up to this multi-vector one lifted the exact score from 0.611 to 0.722 on the same data. The loss from the wrong strategy was bigger than the gain from the better model.
  𝗪𝗵𝘆? 𝗔 𝘀𝘁𝗿𝗼𝗻𝗴 𝗽𝗿𝗲𝗱𝗶𝗰𝘁𝗼𝗿 𝗶𝘀 𝗲𝗺𝗯𝗲𝗱𝗱𝗶𝗻𝗴-𝘀𝗽𝗮𝗰𝗲 𝘀𝗲𝗽𝗮𝗿𝗮𝗯𝗶𝗹𝗶𝘁𝘆 — 𝘄𝗵𝗲𝘁𝗵𝗲𝗿 𝗮 𝗺𝗼𝗱𝗲𝗹'𝘀 𝘁𝗼𝗸𝗲𝗻 𝘃𝗲𝗰𝘁𝗼𝗿𝘀 𝘀𝗶𝘁 𝗳𝗮𝗿 𝗮𝗽𝗮𝗿𝘁 𝗼𝗿 𝗽𝗶𝗹𝗲 𝘁𝗼𝗴𝗲𝘁𝗵𝗲𝗿.
  𝗙𝗼𝗿 𝘀𝗽𝗿𝗲𝗮𝗱 𝗼𝘂𝘁 (𝗝𝗶𝗻𝗮): each token is a precise probe, so TokenANN and MUVERA land on the right document tokens. LEMUR, though, risks collapsing on long-tailed data. 
  𝗙𝗼𝗿 𝗰𝗹𝘂𝘀𝘁𝗲𝗿𝗲𝗱 (𝗔𝗻𝘀𝘄𝗲𝗿𝗔𝗜): one query token pulls back a crowd of close-but-irrelevant tokens, so TokenANN and MUVERA produce weak candidates. LEMUR is the one that still works.
  𝗬𝗼𝘂 𝗰𝗮𝗻 𝗺𝗲𝗮𝘀𝘂𝗿𝗲 𝘀𝗲𝗽𝗮𝗿𝗮𝗯𝗶𝗹𝗶𝘁𝘆 𝗯𝗲𝗳𝗼𝗿𝗲 𝗰𝗼𝗺𝗺𝗶𝘁𝘁𝗶𝗻𝗴. Sample a few hundred token vectors, treat each as a query, and take the standard deviation of their MaxSim scores across documents. Jina sits at 0.157; AnswerAI at 0.050 — at 0.050, nearly every document scores the same, so relevant and irrelevant blur. Recall tracks it: Jina's TokenANN holds Math R@100 of 68.5–88.5% across the four datasets, AnswerAI's 44.6–65.5%.
  𝗦𝗼 𝗯𝗲𝗳𝗼𝗿𝗲 𝘁𝘂𝗻𝗶𝗻𝗴 𝗮𝗻𝘆𝘁𝗵𝗶𝗻𝗴: 
  → Wide spread (closer to 0.15): start with TokenANN or MUVERA 
  → Tight spread (closer to 0.05): start with LEMUR
  The model is only half the decision. Pair it with the wrong strategy, and the best model can't make up the difference.




  [5](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthe-milvus-project_with-the-same-multi-vector-model-and-the-activity-7468323244329050112-Uuyi&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthe-milvus-project_with-the-same-multi-vector-model-and-the-activity-7468323244329050112-Uuyi&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthe-milvus-project_with-the-same-multi-vector-model-and-the-activity-7468323244329050112-Uuyi&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fthe-milvus-project_with-the-same-multi-vector-model-and-the-activity-7468323244329050112-Uuyi&trk=public_post_feed-cta-banner-cta)
* [Riccardo Zanardelli](https://it.linkedin.com/in/riccardozanardelli?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Friccardozanardelli_lossy-vs-lossless-how-the-compression-metaphor-activity-7472233523303616512-tnHc&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Is the success of a [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) model measured by [#accuracy](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Faccuracy&trk=public_post-text), or by the bit-price of its mistakes? 📉
  Let's unpack it!
  In [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text), we often treat errors as a nuisance. But in the world of compression, the ability of a model is defined by one thing: making the residuals cheaper to encode than the original data. Think of it as an honesty audit for machine learning:
  - THE MODEL'S JOB: Identify structure to reduce unpredictability. For example, if we want to predict apartment prices, a simple model can drop the cost of encoding of training dataset vs the cost of encoding the residuals (in our example, see first comment for the link, we move from from 8.318 bits per value down to 6.619 bits)
  - THE SAVING: That 1.70-bit difference is the mathematical proof that the model has "learned" something.
  But here is where the strategy shifts in our toy example:
  1️⃣ THE LOSSLESS SCENARIO (1.18:1 ratio): When we require exact reconstruction, we pay to keep the residuals. This isn't just storing waste; it's keeping the "data alive". These residuals are the "to-do list" for the next model, an option on future understanding. Today’s unexplained "noise" might be tomorrow’s breakthrough signal once a smarter model comes along to claim those bits. However, the model is good enough to make the bit cost of model + residuals < data. 1.18:1 is not terrific, but a 18% bit cost saving indeed!
  2️⃣ THE LOSSY SCENARIO (21:1 ratio): When an approximation is good enough for our goals, a state called satisficing by H.A. Simon (both Nobel and Turing Prize Laurate) we can discard the residuals entirely. By carrying only the model, we achieve a spectacular 21x efficiency gain in storage and speed.
  Choosing between them isn't a ranking; it’s a declaration of purpose. Do you need a perfect archive for future discovery, or a lean tool for immediate action?
  As the source suggest: a model looks spectacular until it has to pay for everything it threw away. Who is paying for your errors?
  [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Compression](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcompression&trk=public_post-text) [#InformationTheory](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Finformationtheory&trk=public_post-text) [#TheFarsightedZipper](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fthefarsightedzipper&trk=public_post-text)
  [https://lnkd.in/duX9a7M6](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FduX9a7M6&urlhash=HBT_&trk=public_post-text)



  [3](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Friccardozanardelli_lossy-vs-lossless-how-the-compression-metaphor-activity-7472233523303616512-tnHc&trk=public_post_social-actions-reactions)







  [3 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Friccardozanardelli_lossy-vs-lossless-how-the-compression-metaphor-activity-7472233523303616512-tnHc&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Friccardozanardelli_lossy-vs-lossless-how-the-compression-metaphor-activity-7472233523303616512-tnHc&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Friccardozanardelli_lossy-vs-lossless-how-the-compression-metaphor-activity-7472233523303616512-tnHc&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Friccardozanardelli_lossy-vs-lossless-how-the-compression-metaphor-activity-7472233523303616512-tnHc&trk=public_post_feed-cta-banner-cta)
* [Nitin Singh](https://in.linkedin.com/in/nitin--singh?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnitin--singh_dsa-100daysofcode-day197-activity-7466364068371587073-Xmt6&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Problem: Number of Connected Components in an Undirected Graph 🌐🔥
  Today’s problem is a classic application of:
  ✅ Union Find (Disjoint Set Union)
  ✅ Connected Components
  ✅ Graph Connectivity
  ✅ Path Compression
  🧠 Problem Summary
  You are given:
  👉 n nodes labeled from 0 to n-1
  👉 An array of undirected edges
  Your task:
  Return the number of connected components in the graph.
  Example:
  n = 5
  0 -- 1
  2 -- 3
  4
  Answer:
  3
  Because there are three separate groups of connected nodes.
  💡 Brute Force Approach
  One way is:
  👉 Build adjacency list
  👉 Run DFS/BFS from every unvisited node
  Every new DFS represents a new component.
  Complexity:
  O(V + E)
  Works well.
  But this problem is a perfect fit for:
  🔥 Union Find
  ⚙️ Union Find Intuition
  Initially:
  Every node belongs to its own component.
  0 1 2 3 4
  Parents:
  0 1 2 3 4
  Each node is its own root.
  When processing an edge:
  0 -- 1
  We merge their sets.
  Now:
  0,1
  belong to the same component.
  Process:
  2 -- 3
  Now:
  2,3
  form another component.
  Result:
  {0,1}
  {2,3}
  {4}
  Total:
  3 components
  🔥 Core Operations
  Find
  Returns the representative (root) of a component.
  find(3)
  might return:
  2
  meaning node 3 belongs to component rooted at 2.
  Union
  Merges two components.
  union(a, b)
  If roots are different:
  rootA != rootB
  combine them into one set.
  ⚡ Path Compression
  During find:
  [self.parent](http://self.parent?trk=public_post-text)[node] = [self.parent](http://self.parent?trk=public_post-text)[[self.parent](http://self.parent?trk=public_post-text)[node]]
  We shorten paths.
  Example:
  Before:
  5 → 4 → 3 → 2 → 1
  After compression:
  5 → 1
  Future lookups become much faster.
  🔥 This is the reason Union Find is nearly O(1) in practice.
  ⚙️ Final Counting Trick
  After processing all edges:
  Find the root of every node.
  [roots.add](http://roots.add?trk=public_post-text)(find(i))
  Every unique root represents:
  One connected component
  Answer:
  len(roots)
  📈 Complexity
  Time Complexity
  Union Find with Path Compression:
  O(E · α(N))
  where:
  α(N)
  is the inverse Ackermann function.
  Practically:
  ≈ O(E)
  Space Complexity
  O(N)
  For parent and size arrays.
  ✨ Why This Problem Is Important
  This problem introduces one of the most powerful graph data structures:
  🔥 Disjoint Set Union (DSU)
  This exact pattern appears in:
  Network connectivity
  Social networks
  Friend circles
  Kruskal’s MST
  Dynamic graph connectivity
  Account merging problems
  The key realization:
  Instead of traversing the graph repeatedly,
  👉 Maintain groups dynamically as edges arrive.
  That is the superpower of Union Find. 🔓🔥
  🔖 [#DSA](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdsa&trk=public_post-text) [#100DaysOfCode](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2F100daysofcode&trk=public_post-text) [#Day197](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fday197&trk=public_post-text) [#Graphs](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fgraphs&trk=public_post-text) [#UnionFind](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Funionfind&trk=public_post-text) [#DSU](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdsu&trk=public_post-text) [#ConnectedComponents](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fconnectedcomponents&trk=public_post-text) [#LeetCode](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fleetcode&trk=public_post-text) [#Algorithms](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Falgorithms&trk=public_post-text) [#ProblemSolving](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fproblemsolving&trk=public_post-text) [#CodingChallenge](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcodingchallenge&trk=public_post-text) [#InterviewPrep](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Finterviewprep&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#SoftwareEngineering](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fsoftwareengineering&trk=public_post-text) [#DeveloperJourney](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdeveloperjourney&trk=public_post-text) [#CodingLife](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcodinglife&trk=public_post-text)




  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnitin--singh_dsa-100daysofcode-day197-activity-7466364068371587073-Xmt6&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnitin--singh_dsa-100daysofcode-day197-activity-7466364068371587073-Xmt6&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fnitin--singh_dsa-100daysofcode-day197-activity-7466364068371587073-Xmt6&trk=public_post_feed-cta-banner-cta)
* [Danish Data Science Community](https://dk.linkedin.com/company/danish-data-science-community?trk=public_post_feed-actor-name)

  6,759 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdanish-data-science-community_elevating-danish-asr-models-a-focus-on-domain-specific-activity-7472903643315458049-vvnn&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Elevating Danish ASR Models: A Focus on Domain-Specific Benchmarking
  The next Danish Data Science Open-Source Standup is fast approaching. The meeting will be held on Thursday, June 18, 2026, at 8:00 PM, where the community will gather to share updates and insights.
  A primary focus of this session will be benchmarking speech-to-text and text-to-speech models using the Coralv3 dataset. This work is critical to ensuring that Danish language models continue to evolve and perform at a high level.
  To make these benchmarks even more robust and applicable to real-world scenarios, the discussion is shifting toward the importance of domain-specific test sets. Efforts are currently underway to secure specialized datasets, which promise to add significant depth to the evaluation process.
  There is a keen interest in expanding this effort further. Specifically, the community is looking for domain-specific test sets, e.g. as those tailored for the medical field. Incorporating such specialized data is essential for testing the versatility and accuracy of ASR models in Danish professional environments.
  Everyone is invited to join the conversation this Thursday to help shape the future of Danish speech technology. For further details and to stay connected with the group, please visit the Slack channel here: [https://lnkd.in/dDgX49Mh](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FdDgX49Mh&urlhash=mbt8&trk=public_post-text)



  [10](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdanish-data-science-community_elevating-danish-asr-models-a-focus-on-domain-specific-activity-7472903643315458049-vvnn&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdanish-data-science-community_elevating-danish-asr-models-a-focus-on-domain-specific-activity-7472903643315458049-vvnn&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdanish-data-science-community_elevating-danish-asr-models-a-focus-on-domain-specific-activity-7472903643315458049-vvnn&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdanish-data-science-community_elevating-danish-asr-models-a-focus-on-domain-specific-activity-7472903643315458049-vvnn&trk=public_post_feed-cta-banner-cta)
* [Finperform](https://uk.linkedin.com/company/finperform-limited?trk=public_post_feed-actor-name)

  27,247 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ffinperform-limited_structured-prereview-of-research-model-proposal-activity-7466990452404011010-4a-i&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Structured PREreview of “RESEARCH MODEL PROPOSAL ON COGNITIVE OVERLOAD ...: The authors conclude that their proposed model applies to Big Data environments, but the data they collected come almost entirely from studies on ... [#bigdata](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fbigdata&trk=public_post-text) [#cdo](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcdo&trk=public_post-text) [#cto](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fcto&trk=public_post-text)



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ffinperform-limited_structured-prereview-of-research-model-proposal-activity-7466990452404011010-4a-i&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ffinperform-limited_structured-prereview-of-research-model-proposal-activity-7466990452404011010-4a-i&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ffinperform-limited_structured-prereview-of-research-model-proposal-activity-7466990452404011010-4a-i&trk=public_post_feed-cta-banner-cta)
* [Ashwat Bijjaragi](https://in.linkedin.com/in/ashwat-bijjaragi-597205317?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashwat-bijjaragi-597205317_pyhon-study-day-9-activity-7467105782174748672-vc42&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  PYHON STUDY DAY - 9
  ===========================================================
  2. float
  ===========================================================
  =>Here 'float' is one of the Pre-Defined Class and Treated as Fundamental Data Type.
  =>The Purpose float data type is that "To Store Real Constant Values OR Floating Point Values " Such as CGPA, Percentage, Percentile...etc
  =>The float data type also used for Representing Scientific Notation of data.
  =>The General Notation of Scientific Notation of data is Given Bellow
  varname= Mantisa e +Exponet
  OR
  varname= Mantisa e -Exponet
  =>The Eqv Floating Point Value for Scientific Notation of data is Given Bellow
  Mantisa x 10 to the power of 'exponent'
  =>The Advantage of Scientific Notation of data is that "To Take Less Memory at time of Storing Biggest Floating Values".
  =>The float data type allows us to decimal Number Values as floating point values But never allows to represent Binary, Octal and Hexa Decimal values.
  ----------------------------------------------------------------------------------------------------------------------------
  Examples
  ----------------------------------------------------------------------------------------------------------------------------
  >>> a=1.2
  >>> print(a,type(a))-----------------------1.2 <class 'float'>
  ---------------------------
  >>> a=0.9
  >>> print(a,type(a))----------------------0.9 <class 'float'>
  ---------------------------------------
  >>> c=a+b
  >>> print(a,type(a),id(a))----------------1.2 <class 'float'> 1767964401776
  >>> print(b,type(b),id(b))----------------2.3 <class 'float'> 1767964408368
  >>> print(c,type(c),id(c))-----------------3.5 <class 'float'> 1767970645264
  ----------------------------------------
  >>> a=10
  >>> b=1.2
  >>> c=a+b
  >>> print(a,type(a),id(a))------------------10 <class 'int'> 140708171859352
  >>> print(b,type(b),id(b))------------------1.2 <class 'float'> 1767970641520
  >>> print(c,type(c),id(c))--------------------11.2 <class 'float'> 1767964408368
  ----------------------------------------------------------------------
  >>> a=3e2
  >>> print(a,type(a))------------------------------300.0 <class 'float'>
  >>> a=10e+2
  >>> print(a,type(a))----------------------------1000.0 <class 'float'>
  >>> a=10e-3
  >>> print(a,type(a))-----------------------------0.01 <class 'float'>
  -------------------------------------------------------------------------------
  >>> a=0.0000000000000000000000000000000000000000000000000000001
  >>> print(a,type(a))
  1e-55 <class 'float'>
  ----------------------------------------------------------------
  >>> a=0b1010.0b1111------------------------------SyntaxError: invalid decimal literal
  >>> a=0o123.0b1111--------------------------------SyntaxError: invalid decimal literal
  >>> a=0xAC.0o12-------------------------------------SyntaxError: invalid decimal literal
  ===============================================================================================



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashwat-bijjaragi-597205317_pyhon-study-day-9-activity-7467105782174748672-vc42&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashwat-bijjaragi-597205317_pyhon-study-day-9-activity-7467105782174748672-vc42&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fashwat-bijjaragi-597205317_pyhon-study-day-9-activity-7467105782174748672-vc42&trk=public_post_feed-cta-banner-cta)
* [Abaidullah Seikh](https://ru.linkedin.com/in/abaidullahseikh?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fabaidullahseikh_k-means-is-one-of-the-most-widely-used-clustering-activity-7470550038277058560-XtLu&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  K-means is one of the most widely used clustering techniques in data science and machine learning. A key part of the algorithm is convergence, the process where cluster centers and point assignments gradually stabilize through repeated updates. Knowing how and why convergence happens helps ensure reliable and meaningful clustering results.
  ✔️ Converges quickly on most data sets, making it efficient for large-scale problems
  ✔️ Offers a simple and interpretable structure for identifying groups
  ✔️ Scales well to large data sets due to its low computational complexity
  ❌ Results depend heavily on initial cluster placement
  ❌ Can misrepresent structure if features are not properly scaled
  ❌ May produce empty or unstable clusters if not configured correctly
  To support consistent convergence:
  🔹 Use k-means++ to start with smarter center positions
  🔹 Apply feature scaling to avoid dominance by larger-scale variables
  🔹 Set suitable values for the iteration limit and convergence threshold
  The image illustrates the K-means convergence process. Data points are assigned to the closest center based on squared distance. After assignment, each center is recalculated as the average of its assigned points. These steps repeat until the center positions no longer change meaningfully.




  [247](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fabaidullahseikh_k-means-is-one-of-the-most-widely-used-clustering-activity-7470550038277058560-XtLu&trk=public_post_social-actions-reactions)







  [1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fabaidullahseikh_k-means-is-one-of-the-most-widely-used-clustering-activity-7470550038277058560-XtLu&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fabaidullahseikh_k-means-is-one-of-the-most-widely-used-clustering-activity-7470550038277058560-XtLu&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fabaidullahseikh_k-means-is-one-of-the-most-widely-used-clustering-activity-7470550038277058560-XtLu&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fabaidullahseikh_k-means-is-one-of-the-most-widely-used-clustering-activity-7470550038277058560-XtLu&trk=public_post_feed-cta-banner-cta)
* [Shemelis Kebede Hundie](https://et.linkedin.com/in/shemelis-kebede-hundie-77743967?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshemelis-kebede-hundie-77743967_really-useful-work-by-dr-merwan-roudane-activity-7471661941556404225-bG2K&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Really useful work by Dr. [Merwan Roudane](https://www.linkedin.com/in/merwan-roudane-41166236b?trk=public_post-text). This new xttestpanel package for Stata fills a clear gap for panel data diagnostics. Worth checking out if you work with fixed effects or random effects models.

  [Merwan Roudane](https://www.linkedin.com/in/merwan-roudane-41166236b?trk=public_post_reshare_feed-actor-name)

  Econometrics and AI

  1mo

  🚀 New on SSC: xttestpanel — A Unified Diagnostic Test Suite for Panel Data Models in Stata
  I am pleased to announce that my new Stata package xttestpanel is now available on SSC.
  xttestpanel is a unified post-estimation diagnostic library for linear panel-data models. It is designed to help applied researchers move beyond the usual limited set of diagnostics and follow a more complete methodological path after estimating fixed-effects, random-effects, pooled, or two-way panel models.
  The package can be used either as a standalone command or directly after xtreg / reghdfe / regress, allowing the researcher to estimate the model once and then run a full battery of diagnostic tests.
  Main diagnostic modules:
  ✅ Heteroskedasticity tests
  • Breusch-Pagan
  • Koenker robust test
  • Juhl & Sosa-Escudero FE test
  • Feng, Li, Tong & Luo two-way FE test
  ✅ Serial-correlation tests
  • Baltagi & Li LM
  • Born-Breitung / Wooldridge robust test
  • Bin Chen (2022) robust portmanteau test
  ✅ Cross-sectional dependence tests
  • Pesaran CD
  • Baltagi-Kao-Peng bias-corrected CD
  • Breusch-Pagan LM
  • Scaled LM
  ✅ Functional-form test
  • Lin, Li & Sun (2014) nonparametric kernel test with wild bootstrap
  ✅ Specification tests
  • Classical Mundlak Hausman
  • Robust weighted Hausman test
  ✅ Multicollinearity diagnostics
  • Within-group VIF
  • Robust VIF based on Ismaeel-Midi-Sani
  The package also includes:
  📌 xttestpanel all to run the whole diagnostic suite
  📌 A combined decision summary
  📌 Optional diagnostic graphs
  📌 Dashboard-style visualization
  📌 Stored results in r() for further reporting
  Installation from SSC:
  ssc install xttestpanel, replace
  Main help file:
  help xttestpanel
  Subcommand help files:
  help xttestpanel\_het
  help xttestpanel\_serial
  help xttestpanel\_csd
  help xttestpanel\_func
  help xttestpanel\_hausman
  help xttestpanel\_vif
  help xttestpanel\_postestimation
  Example:
  xtset id year
  xtreg y x1 x2 x3, fe
  xttestpanel all, dashboard
  Or one diagnostic at a time:
  xttestpanel het, graph
  xttestpanel serial, lags(2)
  xttestpanel csd, graph
  xttestpanel func, reps(299)
  xttestpanel hausman, graph
  xttestpanel vif, graph
  Why xttestpanel?
  Because post-estimation diagnostics in panel data should not stop at one or two classical tests. Applied studies often need to check heteroskedasticity, serial correlation, cross-sectional dependence, functional misspecification, model specification, and multicollinearity before moving to robust inference or alternative estimators.
  xttestpanel provides a structured diagnostic roadmap for panel-data researchers.
  Developed by Dr. Merwan Roudane
  [#Stata](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fstata&trk=public_post_reshare-text) [#Econometrics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Feconometrics&trk=public_post_reshare-text) [#PanelData](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpaneldata&trk=public_post_reshare-text) [#AppliedEconometrics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fappliedeconometrics&trk=public_post_reshare-text) [#PostEstimation](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpostestimation&trk=public_post_reshare-text) [#DiagnosticTests](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdiagnostictests&trk=public_post_reshare-text) [#SSC](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fssc&trk=public_post_reshare-text) [#ResearchMethods](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fresearchmethods&trk=public_post_reshare-text) [#DataAnalysis](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdataanalysis&trk=public_post_reshare-text) [#AcademicResearch](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Facademicresearch&trk=public_post_reshare-text)

  + View C2PA information



  [1](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshemelis-kebede-hundie-77743967_really-useful-work-by-dr-merwan-roudane-activity-7471661941556404225-bG2K&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshemelis-kebede-hundie-77743967_really-useful-work-by-dr-merwan-roudane-activity-7471661941556404225-bG2K&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshemelis-kebede-hundie-77743967_really-useful-work-by-dr-merwan-roudane-activity-7471661941556404225-bG2K&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fshemelis-kebede-hundie-77743967_really-useful-work-by-dr-merwan-roudane-activity-7471661941556404225-bG2K&trk=public_post_feed-cta-banner-cta)
* [Zia Ahmed](https://www.linkedin.com/in/zia-ahmed207?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fzia-ahmed207_healthcare-claims-data-science-activity-7470264023759110144-bGgL&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🏥 Just published an end-to-end Python tutorial on Healthcare Claims Data Analysis
  If you work in health informatics, managed care, or population health analytics, or you're transitioning into healthcare data science, this might be useful.
  The tutorial covers the full analytics pipeline that a data scientist would run inside a Medicare Advantage or value-based care program:
  📋 Patient classification systems: ICD-10, HCC, DRG, Charlson CCI, LACE Index
  🧹 Claims data cleaning & validation (duplicates, financial anomalies, code auditing)
  ⚙️ Feature engineering: PMPM cost, HCC risk scores, utilization metrics
  🏷️ 4-tier risk stratification & chronic disease registries
  📊 Financial risk models: Linear Regression, Random Forest, Gradient Boosting
  🤝 ACO shared savings & risk-adjusted benchmarking
  ✅ HEDIS-like quality measures & care gap identification
  📈 12-panel executive dashboard
  All data is fully synthetic: no PHI, no HIPAA concerns. Everything runs in Jupyter out of the box.
  🔗 [https://lnkd.in/eY3BNRqy](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2FeY3BNRqy&urlhash=pTEA&trk=public_post-text)
  [#HealthcareAnalytics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fhealthcareanalytics&trk=public_post-text) [#DataScience](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fdatascience&trk=public_post-text) [#Python](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpython&trk=public_post-text) [#MedicareAdvantage](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmedicareadvantage&trk=public_post-text) [#ValueBasedCare](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fvaluebasedcare&trk=public_post-text) [#HealthInformatics](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fhealthinformatics&trk=public_post-text) [#MachineLearning](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fmachinelearning&trk=public_post-text) [#PublicHealth](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fhashtag%2Fpublichealth&trk=public_post-text)



  [62](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fzia-ahmed207_healthcare-claims-data-science-activity-7470264023759110144-bGgL&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fzia-ahmed207_healthcare-claims-data-science-activity-7470264023759110144-bGgL&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fzia-ahmed207_healthcare-claims-data-science-activity-7470264023759110144-bGgL&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fzia-ahmed207_healthcare-claims-data-science-activity-7470264023759110144-bGgL&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fzia-ahmed207_healthcare-claims-data-science-activity-7470264023759110144-bGgL&trk=public_post_feed-cta-banner-cta)

39,353 followers

* [1,620 Posts](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fin%2Fadrianolszewski%2Frecent-activity%2F&trk=public_post_follow-posts)
* [11 Articles](https://www.linkedin.com/today/author/adrianolszewski?trk=public_post_follow-articles)

[View Profile](https://pl.linkedin.com/in/adrianolszewski?trk=public_post_follow-view-profile)
[Follow](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7466889448044158976&trk=public_post_follow)

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
