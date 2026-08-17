---
url: https://www.linkedin.com/posts/what-is-network-interference-in-ab-test-share-7459996943780491264-Ewhd/?utm_source=share&utm_medium=member_android&rcm=ACoAACmphIUBo2D4cUQR_ZXSmdR7KMycDh8BUk8
scraped_at: 2026-07-27T14:09:27.770275
depth: 0
---

Understanding Network Interference in A/B Tests | DataInterview.com posted on the topic | LinkedIn



Agree & Join LinkedIn

By clicking Continue to join or sign in, you agree to LinkedIn’s [User Agreement](/legal/user-agreement?trk=linkedin-tc_auth-button_user-agreement), [Privacy Policy](/legal/privacy-policy?trk=linkedin-tc_auth-button_privacy-policy), and [Cookie Policy](/legal/cookie-policy?trk=linkedin-tc_auth-button_cookie-policy).

[Skip to main content](#main-content)


# Understanding Network Interference in A/B Tests

This title was summarized by AI from the post below.

[DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_feed-actor-name)

26,554 followers

2mo

* [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

What is Network Interference? (in A/B test interviews)
👋 Let's learn together ↓
Network interference happens when 𝘁𝗿𝗲𝗮𝘁𝗶𝗻𝗴 𝗼𝗻𝗲 𝘂𝘀𝗲𝗿 𝗮𝗳𝗳𝗲𝗰𝘁𝘀 𝗮𝗻𝗼𝘁𝗵𝗲𝗿 𝘂𝘀𝗲𝗿'𝘀 𝗼𝘂𝘁𝗰𝗼𝗺𝗲.
Standard A/B tests assume independence. But in social networks, marketplaces, or shared resources, users interact. Your treatment group influences your control group through connections.
This breaks SUTVA (Stable Unit Treatment Value Assumption) and makes naive estimates biased.
📐 𝗧𝗵𝗲 𝗽𝗿𝗼𝗯𝗹𝗲𝗺:
Yi(z) = Yi(zi, z-i) ≠ Yi(zi)
Where:
Yi(z) → outcome for unit i under assignment vector z
zi → treatment assigned to unit i
z-i → treatment assigned to all other units
The inequality shows unit i's outcome depends on others' assignments, violating independence.
⚡ 𝗧𝘆𝗽𝗲𝘀 𝗼𝗳 𝗶𝗻𝘁𝗲𝗿𝗳𝗲𝗿𝗲𝗻𝗰𝗲:
① Direct effect: treating unit i changes their own outcome
② Spillover effect: treating unit i changes connected units' outcomes
③ Contamination: control users get exposed through treated neighbors
Real example: you test a referral feature. Treated users invite control users. Control group gets the benefit without the treatment flag.
🎯 𝗛𝗼𝘄 𝘁𝗼 𝗺𝗲𝗮𝘀𝘂𝗿𝗲 𝗶𝘁:
Direct treatment effect = E[Yi(1,0) - Yi(0,0)]
Spillover effect = E[Yi(0,z\_N) - Yi(0,0)]
First isolates individual impact. Second captures peer influence from having treated neighbors z\_N.
🔍 𝗛𝗼𝘄 𝗶𝘀 𝗶𝘁 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁 𝗳𝗿𝗼𝗺 𝗿𝗲𝗴𝘂𝗹𝗮𝗿 𝗔/𝗕 𝘁𝗲𝘀𝘁𝘀?
Regular A/B tests assume treating one user doesn't affect others and randomize at the user level.
Network interference means users influence each other, requires cluster randomization (groups of connected users), and needs special estimators like Horvitz-Thompson to get unbiased effects.
Standard tests give you 30% upward bias from spillover contamination.
✍️ 𝗪𝗵𝗲𝗻 𝘁𝗼 𝘄𝗼𝗿𝗿𝘆 𝗮𝗯𝗼𝘂𝘁 𝗻𝗲𝘁𝘄𝗼𝗿𝗸 𝗶𝗻𝘁𝗲𝗿𝗳𝗲𝗿𝗲𝗻𝗰𝗲:
social features, marketplaces with supply/demand dynamics, shared inventory, pricing changes, or anything where users interact directly or compete for resources.
👉 Land Data & AI jobs on [datainterview.com](https://www.linkedin.com/redir/redirect?url=http%3A%2F%2Fdatainterview%2Ecom&urlhash=bCYW&trk=public_post-text)




[29](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_social-actions-reactions)







[2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_social-actions-comments)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_like-cta)
[Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_comment-cta)



Share

* Copy
* LinkedIn
* Facebook
* X

[DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_comment_actor-name)

2mo

* [Report this comment](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_comment_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=COMMENT&_f=guest-reporting)

👉 Land data, AI, quant jobs on [datainterview.com](http://datainterview.com?trk=public_post_comment-text)

[Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_comment_like)

[Reply](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_comment_reply)

1 Reaction

To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fposts%2Fdatainterview_what-is-network-interference-in-ab-test-activity-7459996969013395456-lj6D&trk=public_post_feed-cta-banner-cta)

## More Relevant Posts

* [MobiusEngine](https://www.linkedin.com/company/mobiusservices?trk=public_post_feed-actor-name)

  95,168 followers

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmobiusservices_here-is-a-pattern-worth-understanding-first-activity-7465160339957923840-oJj3&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Here is a pattern worth understanding.
  First, back-channel referencing is no longer a fringe tactic. By 2026, hiring managers assume the list you provide is curated. They are actively seeking out unlisted former colleagues to get a complete picture. One unresolved conflict from your past can silently disqualify you.
  Second, getting ghosted after a final interview is the new rejection letter. In the current high-volume, fast-paced market, a negative reference check is the most common reason for sudden silence. Recruiters have no incentive to deliver bad news that originates from a third party.
  Third, reference checks are now a data problem. Automated platforms gather and score feedback, meaning a single piece of negative qualitative feedback can translate into a quantitative red flag. You are being judged by an algorithm based on someone else's opinion, often without your knowledge.
  This is the problem [MobiusEngine.ai](https://www.linkedin.com/redir/redirect?url=http%3A%2F%2FMobiusEngine%2Eai&urlhash=TPek&trk=public_post-text) was built for.



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmobiusservices_here-is-a-pattern-worth-understanding-first-activity-7465160339957923840-oJj3&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmobiusservices_here-is-a-pattern-worth-understanding-first-activity-7465160339957923840-oJj3&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fmobiusservices_here-is-a-pattern-worth-understanding-first-activity-7465160339957923840-oJj3&trk=public_post_feed-cta-banner-cta)
* [Tatiana Pridchenko](https://uk.linkedin.com/in/tapridch/en?trk=public_post_feed-actor-name)

  1mo

  Edited

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftapridch_i-think-primary-research-is-overrated-even-activity-7467504056987705344-vL4B&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  I think primary research is overrated (even though it's my bread and butter).
  Let me be clear:
  I'm not saying interviews or surveys or desk research (insert your fav research method) are useless.
  I'm saying we keep piling more of them onto teams that are already drowning in unprocessed data they collected three years ago and never touched again.
  Think of stuff like:
  - The unread Google docs.
  - The unindexed chat logs.
  - The interview recordings nobody ever transcribed.
  - The feedback channel that's been quietly filling up every single day since 2022...
  So, my take is:
  your company probably doesn't need more research.
  You need infrastructure that turns what you already have into something a PM can query in three minutes flat.
  Here's what nobody on the research side wants to hear:
  the value moved.
  It used to live in the act of gathering the data, and now it lives in the architecture that makes that data findable, joinable, and actually usable.
  That's a fundamentally different job.
  If your research team is still being measured by the number of interviews it runs per quarter or slide decks created, then you've got a 2018 research function sitting inside a 2026 product org, and the gap is going to show very soom.
  So stop booking more sessions, and start mapping the mountain of data you've already got.
  The team that wins the next decade won't be the one that asks the most questions.
  It'll be the one that finally listens to everything that's already been said 😮💨
  Do you agree?




  [8](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftapridch_i-think-primary-research-is-overrated-even-activity-7467504056987705344-vL4B&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftapridch_i-think-primary-research-is-overrated-even-activity-7467504056987705344-vL4B&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftapridch_i-think-primary-research-is-overrated-even-activity-7467504056987705344-vL4B&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftapridch_i-think-primary-research-is-overrated-even-activity-7467504056987705344-vL4B&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Ftapridch_i-think-primary-research-is-overrated-even-activity-7467504056987705344-vL4B&trk=public_post_feed-cta-banner-cta)
* [Donato Vaccaro, Ph.D.](https://www.linkedin.com/in/donatovaccaro?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdonatovaccaro_two-independent-research-studies-validate-activity-7466963990703910913-8xyV&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Are you able to discern real from AI generated survey results? [Verasight](https://www.linkedin.com/company/verasight?trk=public_post-text) uses legitimate, verified human respondents to get a true pulse of behaviors and opinions among people in the US. Click below and me know if you have any questions about the Verasight market research panel.

  [Sami Fleischner](https://www.linkedin.com/in/sfleischner?trk=public_post_reshare_feed-actor-name)

  Client Solutions Associate Director @ Verasight | Survey Research

  2mo

  🚀 At [Verasight](https://www.linkedin.com/company/verasight?trk=public_post_reshare-text), high-quality survey research starts with verified human respondents. As concerns around bots, AI-generated responses, and fraudulent survey participation grow, two independent studies of panel providers reinforce the importance of knowing exactly where your data comes from.
  Verasight has taken a different approach:
  • 100% primary data collection (no outsourced sample exchanges)
  • Respondent verification tied to real identities
  • Ongoing quality monitoring throughout the lifecycle of a panelist
  • Recruitment methods designed to reach representative and hard-to-reach populations alike
  Really proud of the work our team has done to build a research platform centered on transparency, verification, and methodological rigor. ⭐
  You can read the full article here: [https://lnkd.in/e88X7Gvi](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fe88X7Gvi&urlhash=mTOP&trk=public_post_reshare-text)?



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdonatovaccaro_two-independent-research-studies-validate-activity-7466963990703910913-8xyV&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdonatovaccaro_two-independent-research-studies-validate-activity-7466963990703910913-8xyV&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdonatovaccaro_two-independent-research-studies-validate-activity-7466963990703910913-8xyV&trk=public_post_feed-cta-banner-cta)
* [Sami Fleischner](https://www.linkedin.com/in/sfleischner?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsfleischner_two-independent-research-studies-validate-activity-7462139118467260417-YKNW&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  🚀 At [Verasight](https://www.linkedin.com/company/verasight?trk=public_post-text), high-quality survey research starts with verified human respondents. As concerns around bots, AI-generated responses, and fraudulent survey participation grow, two independent studies of panel providers reinforce the importance of knowing exactly where your data comes from.
  Verasight has taken a different approach:
  • 100% primary data collection (no outsourced sample exchanges)
  • Respondent verification tied to real identities
  • Ongoing quality monitoring throughout the lifecycle of a panelist
  • Recruitment methods designed to reach representative and hard-to-reach populations alike
  Really proud of the work our team has done to build a research platform centered on transparency, verification, and methodological rigor. ⭐
  You can read the full article here: [https://lnkd.in/e88X7Gvi](https://www.linkedin.com/redir/redirect?url=https%3A%2F%2Flnkd%2Ein%2Fe88X7Gvi&urlhash=mTOP&trk=public_post-text)?



  [15](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsfleischner_two-independent-research-studies-validate-activity-7462139118467260417-YKNW&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsfleischner_two-independent-research-studies-validate-activity-7462139118467260417-YKNW&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsfleischner_two-independent-research-studies-validate-activity-7462139118467260417-YKNW&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsfleischner_two-independent-research-studies-validate-activity-7462139118467260417-YKNW&trk=public_post_feed-cta-banner-cta)
* [Synapse](https://www.linkedin.com/company/synapsehire?trk=public_post_feed-actor-name)

  15,718 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsynapsehire_heres-something-that-keeps-coming-up-in-activity-7470160631975985152-3fMy&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Here's something that keeps coming up in our agency data: treating all 'senior engineer' searches the same is costing you good candidates.
  A senior engineer at a pre-revenue startup looks completely different from one at Google — different signals, different indicators of success, different risk profiles.
  We built our AI screening to take role context into account — the same title means different things depending on company stage, team size, and the actual problems being solved.
  When we tested generic matching vs context-aware matching on 50 real searches, the context-aware model surfaced candidates that the generic model missed in 34% of cases — and those candidates converted to interviews at a 2x higher rate.
  Title + keywords is table stakes. Role context is the differentiator.



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsynapsehire_heres-something-that-keeps-coming-up-in-activity-7470160631975985152-3fMy&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsynapsehire_heres-something-that-keeps-coming-up-in-activity-7470160631975985152-3fMy&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fsynapsehire_heres-something-that-keeps-coming-up-in-activity-7470160631975985152-3fMy&trk=public_post_feed-cta-banner-cta)
* [Campaign Request](https://au.linkedin.com/company/campaign-request?trk=public_post_feed-actor-name)

  1,238 followers

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcampaign-request_middle-of-a-research-project-interviews-activity-7465927357388746752-APcw&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  Middle of a research project. Interviews open. Churn data open. A deadline sitting on a post-it. Four streams converging onto one screen.
  A year ago this would've been week three. Now it's week one.
  The prep used to be most of the work. Transcripts read. Data pulled. Decks compiled. The thinking happened in whatever time was left.
  Now it's done before we sit down. Transcripts summarised. Data graded. Drafts ready.
  Which means basically the whole project is the thinking.
  Can AI technically do the synthesis bit too. Yes. Can it get you to a starting point on a strategy. Also yes.
  Is it the thing that makes the final call. The one that knows whether this lands with your audience, sits inside the brand, and actually moves a number. No.
  That part still happens at the desk.
  The marketers doing the best work in 2026 are getting good at curating and strategising. And it's a much better job when that's the whole job.
  Strategy used to be the thing you ran out of time for. Now it's the main chunk of work.
  That's the part most of us got into this for.




  [3](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcampaign-request_middle-of-a-research-project-interviews-activity-7465927357388746752-APcw&trk=public_post_social-actions-reactions)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcampaign-request_middle-of-a-research-project-interviews-activity-7465927357388746752-APcw&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcampaign-request_middle-of-a-research-project-interviews-activity-7465927357388746752-APcw&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fcampaign-request_middle-of-a-research-project-interviews-activity-7465927357388746752-APcw&trk=public_post_feed-cta-banner-cta)
* [Clayton Newman](https://www.linkedin.com/in/clayton-newman?trk=public_post_feed-actor-name)

  1mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fclayton-newman_what-would-you-do-an-employer-invites-activity-7468073142964572161-a27D&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  What would you do?
  An employer invites you to participate in an offsite employee sentiment lab. The lab asks you to wear a noninvasive neural monitor (they look like ear buds). And they explain they are able to measure your brain's response to the questions asked. Whether you respond verbally or not.
  The vendor conducting the sentiment analysis explains your neural rights. Your individual neural information is contractually prevented from being shared with your employer. It is stored in the cloud protected by industry best practice security. And finally, the neural data will be deleted after the study is completed, but no longer than within 3 months.
  Would you give the lab tech permission to measure you and put on the neural device?
  What if during the sentiment analysis they ask you what is your level of support to commit fraud? To commit violence at work? To violate the company's social media policy by leaking company internal communications? And follow up by asking which factors you agree would justify committing each of the above?
  Would you stay?
  What if a new employer required you to take the neural test prior to employment? Would you accept the job offer?
  What would you do if the purchase of consumer ear buds legally granted the company consent to measure your neural responses to advertising, images, and content you interact with on your phone? Would you buy the ear buds?
  This is coming, so proactively reflecting on how to best approach these issues is worthwhile.



  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fclayton-newman_what-would-you-do-an-employer-invites-activity-7468073142964572161-a27D&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fclayton-newman_what-would-you-do-an-employer-invites-activity-7468073142964572161-a27D&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fclayton-newman_what-would-you-do-an-employer-invites-activity-7468073142964572161-a27D&trk=public_post_feed-cta-banner-cta)
* [DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_feed-actor-name)

  26,554 followers

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-selection-bias-in-ab-test-interviews-activity-7462170438731943936-xPQ9&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  What is Selection Bias? (in A/B test interviews)
  👋 Let's learn together ↓
  𝗦𝗲𝗹𝗲𝗰𝘁𝗶𝗼𝗻 𝗯𝗶𝗮𝘀 𝗼𝗰𝗰𝘂𝗿𝘀 𝘄𝗵𝗲𝗻 𝘆𝗼𝘂𝗿 𝘀𝗮𝗺𝗽𝗹𝗲 𝘀𝘆𝘀𝘁𝗲𝗺𝗮𝘁𝗶𝗰𝗮𝗹𝗹𝘆 𝗱𝗶𝗳𝗳𝗲𝗿𝘀 𝗳𝗿𝗼𝗺 𝘁𝗵𝗲 𝗽𝗼𝗽𝘂𝗹𝗮𝘁𝗶𝗼𝗻.
  Your estimates become wrong even if your model is perfect. The data you observe doesn't represent the reality you care about.
  Example: if only high-performing users complete your survey, their average satisfaction will be higher than the true population mean. That gap is selection bias.
  📐 𝗧𝗵𝗲 𝗺𝗮𝘁𝗵:
  Bias = E[θ̂ₛ] - θ = E[θ | S = 1] - E[θ]
  Where:
  θ̂ₛ → estimate from selected sample
  θ → true population parameter
  S = 1 → indicator that unit was selected
  E[θ | S = 1] → expected value in selected sample
  ⚡ 𝗛𝗼𝘄 𝗶𝘁 𝗵𝗮𝗽𝗽𝗲𝗻𝘀:
  ① Sample selection depends on outcome or related variables
  ② Observed distribution shifts away from population
  ③ Estimates calculated on biased sample
  ④ Results don't generalize to target population
  Common causes: non-random sampling, survivorship (only seeing successes), missing data that's not random, conditioning on a collider variable.
  🔍 𝗛𝗼𝘄 𝗶𝘀 𝗶𝘁 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁 𝗳𝗿𝗼𝗺 𝗦𝗮𝗺𝗽𝗹𝗶𝗻𝗴 𝗘𝗿𝗿𝗼𝗿?
  Sampling error is random variation from taking a sample. It decreases with larger samples and averages out to zero.
  Selection bias is systematic. It doesn't disappear with more data. Your sample is fundamentally unrepresentative, so adding more biased observations just gives you more confident wrong answers.
  🧮 𝗖𝗼𝗿𝗿𝗲𝗰𝘁𝗶𝗼𝗻 𝗺𝗲𝘁𝗵𝗼𝗱𝘀:
  Heckman correction models the selection mechanism first, then adjusts outcome estimates.
  Inverse probability weighting reweights observations by 1/P(selected) to recover population distribution.
  Randomized designs with intent-to-treat analysis prevent selection by design.
  ✍️ 𝗪𝗵𝗲𝗻 𝘁𝗼 𝘄𝗮𝘁𝗰𝗵 𝗳𝗼𝗿 𝗦𝗲𝗹𝗲𝗰𝘁𝗶𝗼𝗻 𝗕𝗶𝗮𝘀:
  whenever participation is voluntary, data is missing non-randomly, or you're analyzing survivors (customers who didn't churn, products still in market, experiments that finished).
  👉 Land Data & AI jobs on [datainterview.com](https://www.linkedin.com/redir/redirect?url=http%3A%2F%2Fdatainterview%2Ecom&urlhash=bCYW&trk=public_post-text)




  [73](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-selection-bias-in-ab-test-interviews-activity-7462170438731943936-xPQ9&trk=public_post_social-actions-reactions)







  [1 Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-selection-bias-in-ab-test-interviews-activity-7462170438731943936-xPQ9&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-selection-bias-in-ab-test-interviews-activity-7462170438731943936-xPQ9&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-selection-bias-in-ab-test-interviews-activity-7462170438731943936-xPQ9&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-selection-bias-in-ab-test-interviews-activity-7462170438731943936-xPQ9&trk=public_post_feed-cta-banner-cta)
* [DataInterview.com](https://www.linkedin.com/company/datainterview?trk=public_post_feed-actor-name)

  26,554 followers

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-difference-in-differences-in-causal-activity-7464707719476695040-nPm0&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  What is Difference-in-Differences? (in causal inference interviews)
  👋 Let's learn together ↓
  DiD is a 𝗾𝘂𝗮𝘀𝗶-𝗲𝘅𝗽𝗲𝗿𝗶𝗺𝗲𝗻𝘁𝗮𝗹 𝗺𝗲𝘁𝗵𝗼𝗱 that estimates causal treatment effects by comparing outcome changes across treated and control groups over time.
  You measure how much the treatment group changed before and after treatment. Then subtract how much the control group changed over the same period. The difference of these differences is your causal estimate.
  This removes time trends that affect both groups equally.
  📐 𝗧𝗵𝗲 𝗲𝘀𝘁𝗶𝗺𝗮𝘁𝗼𝗿:
  δ\_DiD = (Ȳ\_T,post - Ȳ\_T,pre) - (Ȳ\_C,post - Ȳ\_C,pre)
  Where:
  Ȳ\_T,post → average outcome for treated group after treatment
  Ȳ\_T,pre → average outcome for treated group before treatment
  Ȳ\_C,post → average outcome for control group after treatment
  Ȳ\_C,pre → average outcome for control group before treatment
  ⚡ 𝗛𝗼𝘄 𝗶𝘁 𝘄𝗼𝗿𝗸𝘀:
  ① Identify a treatment group that receives an intervention at time t
  ② Find a control group that doesn't receive treatment but follows similar trends
  ③ Measure outcomes for both groups before and after treatment
  ④ Calculate the difference in changes between groups
  ⑤ The result is your ATT (average treatment effect on the treated)
  You can also run this as a regression with an interaction term between treatment indicator and time indicator. The coefficient on that interaction is your DiD estimate.
  🧐 𝗛𝗼𝘄 𝗶𝘀 𝗶𝘁 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁 𝗳𝗿𝗼𝗺 𝗔/𝗕 𝘁𝗲𝘀𝘁𝗶𝗻𝗴?
  A/B testing randomly assigns treatment, so groups are identical on average. You just compare post-treatment outcomes.
  DiD uses observational data where treatment isn't random. You need pre-treatment data to control for baseline differences. The key assumption is parallel trends: both groups would've moved together without treatment.
  A/B tests give you ATE (average treatment effect). DiD gives you ATT and only works if trends would've stayed parallel.
  ✍️ 𝗪𝗵𝗲𝗻 𝘁𝗼 𝘂𝘀𝗲 𝗗𝗶𝗗:
  when you can't randomize treatment but have pre-treatment data and a control group that shares similar trends. Common in policy evaluation, marketing rollouts, and feature launches by region.
  👉 Land Data & AI jobs on [datainterview.com](https://www.linkedin.com/redir/redirect?url=http%3A%2F%2Fdatainterview%2Ecom&urlhash=bCYW&trk=public_post-text)




  [77](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-difference-in-differences-in-causal-activity-7464707719476695040-nPm0&trk=public_post_social-actions-reactions)







  [2 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-difference-in-differences-in-causal-activity-7464707719476695040-nPm0&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-difference-in-differences-in-causal-activity-7464707719476695040-nPm0&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-difference-in-differences-in-causal-activity-7464707719476695040-nPm0&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fdatainterview_what-is-difference-in-differences-in-causal-activity-7464707719476695040-nPm0&trk=public_post_feed-cta-banner-cta)
* [Sean Jecko](https://www.linkedin.com/in/seanjecko?trk=public_post_feed-actor-name)

  2mo

  + [Report this post](/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fseanjecko_im-watching-qual-deteriorate-in-real-time-activity-7461067938113003520-0sHG&trk=public_post_ellipsis-menu-semaphore-sign-in-redirect&guestReportContentType=POST&_f=guest-reporting)

  I'm watching qual deteriorate in real time. Here's what it looks like from the inside.
  Three things I'm seeing right now, from inside active fieldwork.
  Cheap talent is replacing experience. I've had research firms tell me in no uncertain terms: "A college grad can take respondents through a discussion guide, so why would we pay for an experienced moderator?" This is how you get technically correct sessions that reveal nothing. It's the analog version of slop. Moderation is more than question delivery. It's knowing what just happened in the session and deciding in real time what to do next.
  Qual at scale is an oxymoron. Qual was never designed to scale. The intimacy, the probing, the ability to follow an unexpected thread -- these things don't survive supersizing. If you're running qual at scale and feel good about it, you're not doing qual anymore. You're doing something else and calling it qual.
  Fraud in B2B qual is the worst I've ever seen. People posing as SMEs, doing a convincing enough job to pass screeners. These interviews produce data (mostly respondents riffing on AI responses), and the data looks...well...like data. But it's data gathered from respondents who aren't real. In B2B qual especially, where you're paying for access to genuine expertise, this is a serious problem that almost nobody is talking about openly.
  None of this is inevitable. But it's where the incentives are pointing right now.
  As a consultant, I'd say it like this: your system is perfectly designed to deliver the results you're getting. If you want different results, change the system.
  If you're reading this, you're in the system. And you can change it.



  [197](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fseanjecko_im-watching-qual-deteriorate-in-real-time-activity-7461067938113003520-0sHG&trk=public_post_social-actions-reactions)







  [44 Comments](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fseanjecko_im-watching-qual-deteriorate-in-real-time-activity-7461067938113003520-0sHG&trk=public_post_social-actions-comments)

  [Like](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fseanjecko_im-watching-qual-deteriorate-in-real-time-activity-7461067938113003520-0sHG&trk=public_post_like-cta)
  [Comment](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fseanjecko_im-watching-qual-deteriorate-in-real-time-activity-7461067938113003520-0sHG&trk=public_post_comment-cta)



  Share
  + Copy
  + LinkedIn
  + Facebook
  + X

  To view or add a comment, [sign in](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fseanjecko_im-watching-qual-deteriorate-in-real-time-activity-7461067938113003520-0sHG&trk=public_post_feed-cta-banner-cta)

26,554 followers

[View Profile](https://www.linkedin.com/company/datainterview?trk=public_post_follow-view-profile)
[Connect](https://www.linkedin.com/signup/cold-join?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Ffeed%2Fupdate%2Furn%3Ali%3Aactivity%3A7459996969013395456&trk=public_post_follow)

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
