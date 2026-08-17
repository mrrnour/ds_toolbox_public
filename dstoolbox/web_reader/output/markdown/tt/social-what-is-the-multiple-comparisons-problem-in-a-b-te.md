---
url: https://www.linkedin.com/posts/what-is-the-multiple-comparisons-problem-share-7463619959332528128-AmFo/?utm_source=share&utm_medium=member_android&rcm=ACoAACmphIUBo2D4cUQR_ZXSmdR7KMycDh8BUk8
scraped_at: 2026-07-27T14:10:25.909585
depth: 0
---

What is the Multiple Comparisons Problem? (in A/B test interviews)
👋 Let's learn together ↓
The Multiple Comparisons Problem happens when you run many statistical tests at once, inflating your… | DataInterview.com



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# DataInterview.com’s Post

[DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_feed-actor-name)

26,554 followers

2mo

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

What is the Multiple Comparisons Problem? (in A/B test interviews)
👋 Let's learn together ↓
The Multiple Comparisons Problem happens when you run many statistical tests at once, inflating your false positive rate far beyond your chosen significance level.
Run 100 tests at α = 0.05? You'll get about 5 false positives by pure chance, even when nothing is real. Your "discoveries" become noise.
This is why you can't just test 50 metrics and celebrate whatever looks significant.
📐 𝗧𝗵𝗲 𝗺𝗮𝘁𝗵:
P(≥1 false positive) = 1 - (1 - α)^m
Where:
α → significance level per test (usually 0.05)
m → number of independent tests
Result → family-wise error rate (FWER)
Example: 20 tests at α=0.05 gives 64% chance of at least one false positive. 100 tests? 99.4%.
⚡ 𝗪𝗵𝘆 𝗶𝘁 𝗵𝗮𝗽𝗽𝗲𝗻𝘀:
① You test multiple metrics in one experiment
② Each test has a 5% false positive rate
③ More tests = more chances to get lucky
④ One "significant" result feels like a win, but it's probably random
🔍 𝗛𝗼𝘄 𝘁𝗼 𝗳𝗶𝘅 𝗶𝘁:
𝗕𝗼𝗻𝗳𝗲𝗿𝗿𝗼𝗻𝗶 𝗖𝗼𝗿𝗿𝗲𝗰𝘁𝗶𝗼𝗻:
Divide your α by m. Test at α/m instead of α.
Simple and safe, but loses power when m is large.
𝗕𝗲𝗻𝗷𝗮𝗺𝗶𝗻𝗶-𝗛𝗼𝗰𝗵𝗯𝗲𝗿𝗴 (𝗙𝗗𝗥):
Control the false discovery rate instead of FWER.
Rank p-values, reject where p(k) ≤ k/m × q.
More powerful than Bonferroni, better for large m.
🧐 𝗙𝗪𝗘𝗥 𝘃𝘀 𝗙𝗗𝗥:
FWER controls the probability of any false positive. Strict. Use when false positives are expensive.
FDR controls the expected proportion of false positives among your discoveries. Relaxed. Use when you can tolerate some noise and want more power.
✍️ 𝗪𝗵𝗲𝗻 𝘁𝗼 𝘄𝗼𝗿𝗿𝘆 𝗮𝗯𝗼𝘂𝘁 𝘁𝗵𝗶𝘀:
A/B testing many metrics, genome studies with thousands of variants, feature selection with hundreds of candidates, or any time you're fishing for significance across multiple tests.
👉 Land Data & AI jobs on [datainterview.com](https://www.linkedin.com/redir/redirect?url=http%3A%2F%2Fdatainterview%2Ecom&urlhash=bCYW&trk=public_post-text)




[45](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_social-actions-reactions)







[1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_social-actions-comments)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

[DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_comment_actor-name)

2mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

👉 Land data, AI, quant jobs on [datainterview.com](http://datainterview.com?trk=public_post_comment-text)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_comment_reply)
[1 Reaction](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_comment_reactions)

2 Reactions

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-the-multiple-comparisons-problem-activity-7463619982241918978-HN2Q&trk=public_post_feed-cta-banner-cta)

26,554 followers

[View Profile](https://www.linkedin.com/company/datainterview?trk=public_post_follow-view-profile)
[Follow](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7463619982241918978&trk=public_post_follow)

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
