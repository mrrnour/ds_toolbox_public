---
url: https://discord.com/blog/measure-less-to-learn-more-using-fewer-higher-quality-metrics-to-capture-what-matters
scraped_at: 2026-07-27T14:10:37.586802
depth: 0
---

Measure Less to Learn More: Using Fewer, Higher-quality Metrics to Capture What Matters

[Lorem ipsum dolor sit

>](#)[Mi neque maecenas](#)

Engineering & Developers

# Measure Less to Learn More: Using Fewer, Higher-quality Metrics to Capture What Matters

Jake Mainwaring

April 24, 2026

If you’re reading this blog post, you’re likely familiar with the pull toward more metrics. As organizations grow, so too does the list of things people want to measure. Different metrics matter for different teams, and everyone has Metrics FOMO, worried that leaving one out could prevent us from reaching our Next Big Insight.

At Discord, this happened with our Default Metric List: a set of metrics that are automatically included in every experiment. Over time, that default list grew as teams added metrics they cared about, while few were removed. We took a step back and asked if we might be better off measuring *less*.

To data teams, suggesting we measure less feels like heresy. “Our job is to measure! Why would we, the organization’s shrewdest pattern finders, knowingly leave data on the table?” The encounter below might look familiar:

This urge is real, but having *too many* metrics brings a new set of issues. Beyond higher compute costs and a harder time navigating experiment readouts, having more metrics highlights an inherent tradeoff:

* **Leaving** [**p-values**](https://pmc.ncbi.nlm.nih.gov/articles/PMC2895822/) **as-is** has the potential for too many false positives. For example, if you have 100 metrics and set a 5% p-value threshold for statistical significance, 5 of your metrics are going to be statistically significant *just by random chance*.
* **Adjusting p-values using a multiple hypothesis correction** can result in fewer false positives, but worse recall in detecting real changes. In this situation, ”Recall” is defined as the proportion of true positives that we catch.

In this article, we explore our journey to address this issue and show that there is no One Fancy Statistical Method™️ to get around this. **The best solution is to use fewer, high-quality metrics that capture distinct concepts.**

## **The Multiple Comparisons Problem**

In Discord's experiments, we apply a Benjamini-Hochberg (BH) correction to control false discovery rates. BH is one of many approaches to handle the multiple comparisons [problem](https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2008/1.5_Bonferroni_FDR.pdf). As more metrics are added to an experiment, the likelihood of a false alarm increases, meaning higher likelihood of at least one metric being flagged as significant by chance alone.

Benjamini-Hochberg keeps the false discovery rate (FDR) at or below 5%, regardless of how many metrics are in the pool. It does this by making individual metrics harder to flag as statistically-significant.

In the following example, the metric with an unadjusted p-value of 0.038 would be statistically significant when left as is, but not when its p-value is adjusted:

BH ranks metrics by their p-values in ascending order, as seen on the x-axis labeled “metric rank.” It then compares each p-value against a threshold that increases with rank. For each metric, this threshold is *i* × *α* / *n*, where *i* = rank, *α* = significance level (0.05), and *n* = number of metrics. A metric is flagged as significant if its p-value falls below its rank-specific threshold, indicated by the sloped, dashed line.

Without prior knowledge of which metrics are likely to move, BH treats all metrics the same. It has no way to allocate stricter or looser thresholds based on how likely each metric is to reflect a real change. Bayesian methods could help here, but we aren’t opening that can of worms today. (Although the team is fond of Bayesian statistics, our default statistics engine is frequentist. More on Bayesian approaches below!)

Benjamini-Hochberg keeps the false discovery rate low by making individual metrics harder to flag, but this comes at a cost to recall. In other words, we might be over-correcting and concealing *real* movements. With p-value adjustments, false alarms become less common (woo!), but genuine changes are harder to detect (boo!).

**The best way to improve recall without causing too many false alarms is by analyzing fewer metrics.**

## **Seeing for Ourselves**

Through most of its history, statistics has been taught using closed-form formulas derived from probability theory. Perfect reading material when you want to fall asleep at night. Lucky for us, it's now easy to run simulations and see how things actually unfold under different scenarios. Rather than taking statistical theory at face value, we wanted to see for ourselves how these numbers play out.

In our case, we simulated 50,000 experiments with a known effect and a fixed number of metrics. For each of 20 metrics, we drew a random noise value from a normal distribution centered around zero (μ = 0, σ = 1) to capture natural variation. One metric has a true effect of *z = 2.8* (-5.2%), which matches a real change observed in a past experiment. For that metric, we drew from a normal distribution centered around 2.8 and added similar noise:

We also ran simulations across different metric counts to understand the relationship between the number of metrics in an experiment and how that impacts the false alarm/recall tradeoff. For each simulated experiment, using the typical p-value threshold of α = 0.05, we can answer:

1. **Did any null metric falsely flag?** Or, were any of the first 20 "no effect" metrics' p-values less than 0.05?
2. **Did the real effect get flagged?** In this case, the ”real effect” metric’s adjusted p-value is less than 0.05.

Below is the false alarm rate and recall across different numbers of metrics. This is also based on 50,000 simulated experiments, where one metric has a real effect, and the remaining metrics do not.

There is a clear pattern: more metrics in the experiment means a stricter correction needs to be made, leading to worse recall under BH. In addition, the uncorrected false alarm rate grows increasingly high with more metrics.

Reducing the number of metrics that get automatically added to every experiment puts us in a stronger position on both fronts.

## **Choosing Metrics for Removal**

In 2024, we first started implementing our “less is more” metric strategy by standardizing based on 7-day lookback windows (7d). This helped clean up the different windows across 1-day, 14-day, and 30-day timeframes and was a step in the right direction, but the core problem remained: we needed to cut back on metrics measuring overlapping behaviors. This raises the question: **which metrics should be removed?**

To figure that out, we first calculated treatment effect correlations across our recent experiments to see which metrics tended to move in a similar direction across experiments.

Below is an example with eight illustrative metrics:

A few of these pairs, such as *metric\_one and metric\_four*, are highly correlated, which is common when metrics measure related concepts. They’re good candidates for consolidation without losing meaningful signal, as consolidating here benefits every other metric in the pool. Fewer metrics means a less aggressive BH adjustment, making it easier to detect real effects in the metrics we do include.

Correlations tell us which pairs of metrics move together, but what we really want to know is how redundant the full set of metrics is. How many truly independent things are we measuring?

We found that Principal Component Analysis ([PCA](https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/)) can be a helpful tool here. Much has been written about PCA, but at a high level, PCA can help us reduce dimensionality and find the directions in which data varies the most. If we have two metrics that largely move together, PCA will show that most of the variation can be captured when projecting onto a single axis:

When running Principal Component Analysis on our historic experiment data, we found that a large proportion of variance (y-axis) was captured by only a few components:

This strengthened our original hypothesis: many of our engagement-related metrics, for example, collapsed onto one component, suggesting they measured a similar concept.

Metric correlations and PCA did not tell us exactly *what* to cut, but they surfaced redundant metrics for discussion with owning teams, who added business context to inform which ones to keep. These findings reassured us that many of these metrics could be removed without a substantial loss of signal of what’s important to the organization. The balance between coverage and recall is hard to quantify, but these findings provided confidence to move toward fewer, higher-quality metrics.

## **Moving Forward**

Of course, the journey here is never over. We’ve been exploring ways to push this work forward and have a few approaches in mind:

### **Empirical Bayes**

While teams can already analyze experiments with Bayesian methods, our internal tooling defaults to using uninformative priors. We’re looking into [using Empirical Bayes](https://efron.ckirby.su.domains/papers/2021EB-concepts-methods.pdf) to help estimate more informative priors from past data, assigning higher prior probability to metrics that have historically shown real effects, raising recall without inflating false discovery rates.

### **Automated redundancy detection**

Rather than periodic manual audits, we could consider automating the analyses above to flag metrics that have become redundant as behavior evolves, keeping our overall pool lean as we go.

### **Further consolidation**

There’s room to consolidate even further by using composite measures, the idea behind an "Overall Evaluation Criterion" (OEC) as described in Chapter 7 of [*Trustworthy Online Controlled Experiments*](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) (Kohavi, Tang, Xu). With a small enough number of default metrics, we could eventually drop p-value adjustments altogether, leading to even better recall.

All told, we were able to cut our default set of metrics from **~50 to ~15** by collapsing platform-level breakouts into parent metrics and removing engagement metrics that were largely measuring the same thing. This improved **our ability to catch a real, moderate-sized effect by ~45%!**

We hope our experience here serves as a reminder to all that casting a wider net comes at a cost. Teams should aim to use the smallest number of metrics to capture what matters. In a time when adding *more* becomes increasingly easy—more metrics, more lines of code, more words—there’s value in choosing what *not* to measure.

If you’d like to read more engineering stories like this, explore the [Engineering & Developers section](https://discord.com/category/engineering) of the Discord Blog! Or, if you want to help us tackle some of these challenges, we’d love to have you join us. [Explore our Careers page](https://discord.com/careers) periodically, as openings pop up all the time!

Tags

No items found.

Jake Mainwaring

Senior Data Engineer at Discord. Focused on experimentation, metrics pipelines, applied ML, and causal inference methods.‍

## related articles

[Product & Features

### Discord Patch Notes: July 7, 2026](/blog/discord-patch-notes-july-7-2026)

[Product & Features

### Discord is Now on Meta Quest: Reach Out to Your Servers While in VR](/blog/discord-is-now-on-meta-quest-reach-out-to-your-servers-while-in-vr)

[Product & Features

### Discord Update: June 25, 2026 Changelog](/blog/discord-update-june-25-2026-changelog)

[Product & Features

### Introducing: You Bar](/blog/you-bar)

[Product & Features

### Discord Patch Notes: June 4, 2026](/blog/discord-patch-notes-june-4-2026)

[Product & Features

### Official Discord Integrations for Steal a Brainrot, Grow a Garden, Brookhaven RP, and more](/blog/official-discord-integrations-for-steal-a-brainrot-grow-a-garden-brookhaven-rp-and-more)

[Product & Features

### Making It Easier Than Ever to Connect with Friends in League & VAL!](/blog/making-it-easier-than-ever-to-connect-with-friends-in-league-val)

[Product & Features

### Every Voice and Video Call on Discord Is Now End-to-End Encrypted](/blog/every-voice-and-video-call-on-discord-is-now-end-to-end-encrypted)

[Product & Features

### Nitro Now Comes with Xbox Game Pass and New Benefits. Welcome to Nitro Rewards.](/blog/nitro-rewards)

[Product & Features

### Stock Up in the New Rust Shop! Enjoy a Discord-Only 20% Sale on Most Items until 5/21](/blog/rust-shop-on-discord-launch-sale)

[Product & Features

### Discord Patch Notes: April 6, 2026](/blog/discord-patch-notes-april-6-2026)

[Product & Features

### Discord Update: March 24, 2026 Changelog](/blog/discord-update-march-24-2026-changelog)

[Product & Features

### Discord Patch Notes: March 6, 2026](/blog/discord-patch-notes-march-6-2026)

[Product & Features

### How to Change Your Theme to Bring Your Vibe to Discord](/blog/bring-your-vibe-to-discord-with-new-themes-in-nitro)

[Product & Features

### Discord Patch Notes: February 4, 2026](/blog/discord-patch-notes-february-4-2026)

[Product & Features

### Gift Ideas for the Dedicated Discord User in Your Life](/blog/gift-ideas-for-the-discord-user)

[Product & Features

### Your Discord Checkpoint is Rolling Out! Celebrate What You Did in 2025](/blog/checkpoint-2025-discord-year-in-review)

[Product & Features

### Save and Display Your Faves: Add Discord Shop & Marvel Rivals Items to Your Profile’s Wishlist](/blog/save-and-display-your-faves-add-discord-shop-marvel-rivals-items-to-your-profiles-wishlist)

[Product & Features

### Bringing In-Game Commerce to Discord Communities](/blog/bringing-in-game-commerce-to-discord-communities)

[Product & Features

### Discord Update: November 6, 2025 Changelog](/blog/discord-update-november-6-2025-changelog)

[Product & Features

### A Cornucopia of Updates Make Discord on Desktop Fresher Than a Crisp Fall Breeze](/blog/a-cornucopia-of-updates-make-discord-on-desktop-fresher-than-a-crisp-fall-breeze)

[Product & Features

### Discord Patch Notes: November 4, 2025](/blog/discord-patch-notes-november-4-2025)

[Product & Features

### Discord Patch Notes: October 7, 2025](/blog/discord-patch-notes-october-7-2025)

[Product & Features

### Discord Update: September 25, 2025 Changelog](/blog/discord-update-september-25-2025-changelog)

[Product & Features

### New Looks for Nitro, New Looks for You. Get Yourself a Nitro-exclusive Profile Bundle!](/blog/new-looks-for-nitro-new-looks-for-you-get-yourself-a-nitro-exclusive-profile-bundle)

[Product & Features

### Transforming Game Discovery with Instant Play Experiences on Discord](/blog/transforming-game-discovery-with-instant-play-experiences-on-discord)

[Product & Features

### Reward Your Play: Complete Quests. Earn Orbs. Get Sweet Stuff.](/blog/discord-orbs)

[Product & Features

### Discord Update: June 30, 2025 Changelog](/blog/discord-update-june-30-2025-changelog)

[Product & Features

### Get More From Your Boosts With New Server Perks](/blog/get-more-from-your-boosts-with-new-server-perks)

[Product & Features

### Gift Nitro and Earn A Flavorful Splash for your Avatar](/blog/gift-nitro-and-earn-a-flavorful-splash-for-your-avatar)

[Product & Features

### Discord Social SDK Updates & Integrations](/blog/discord-social-sdk-updates-integrations)

[Product & Features

### Discord Patch Notes: June 3, 2025](/blog/discord-patch-notes-june-3-2025)

[Product & Features

### Go Beyond, Plus Ultra! with the My Hero Academia Collection](/blog/go-beyond-plus-ultra-with-the-my-hero-academia-collection)

[Product & Features

### STAR WARS™ Makes Its Way to Discord](/blog/star-wars-makes-its-way-to-discord)

[Product & Features

### Discord Patch Notes: May 1, 2025](/blog/discord-patch-notes-may-1-2025)

[Product & Features

### Worthy of a Plaque: Nameplates Land in the Shop](/blog/nameplates-land-in-the-shop)

[Product & Features

### Make More Closet Space! Nitro Members Can Now Keep Avatar Decoration Quest Rewards for Longer](/blog/nitro-members-keep-quest-rewards-longer)

[Product & Features

### Discord Patch Notes: April 3, 2025](/blog/discord-patch-notes-april-3-2025)

[Product & Features

### Discord Update: March 25, 2025 Changelog](/blog/discord-update-march-25-2025-changelog)

[Product & Features

### Revamped Overlay & Refreshed Desktop Give Game Time a Boost](/blog/player-release-q12025)

[Product & Features

### Discord Patch Notes: March 11, 2025](/blog/discord-patch-notes-march-11-2025)

[Product & Features

### Discord Patch Notes: February 3, 2025](/blog/discord-patch-notes-february-3-2025)

[Product & Features

### Discord Update: December 19, 2024 Changelog](/blog/discord-update-december-19-2024-changelog)

[Product & Features

### Discord Patch Notes: December 5, 2024](/blog/discord-patch-notes-december-5-2024)

[Product & Features

### Discord Update: November 18, 2024 Changelog](/blog/discord-update-november-18-2024-changelog)

[Product & Features

### Celebrate Arcane’s Second Season with a new Shop Collection](/blog/arcane-shop-collection)

[Product & Features

### Discord Patch Notes: November 1, 2024](/blog/discord-patch-notes-november-1-2024)

[Product & Features

### Set Out for a Discord Adventure! Check Out Our Roll20 Adventure & D&D Shop Collection](/blog/discord-roll20-adventure-and-dnd-shop-collection)

[Product & Features

### Discord Patch Notes: October 1, 2024](/blog/discord-patch-notes-october-1-2024)

[Product & Features

### Discord Update: September 26, 2024 Changelog](/blog/discord-update-september-26-2024-changelog)

[Product & Features

### Discover More Ways to Play with Apps – Now Anywhere on Discord!](/blog/discover-more-ways-to-play-with-apps-now-anywhere-on-discord)

[Product & Features

### Legacy Shop Favorites Emerge from The Vault for a First Anniversary Encore!](/blog/legacy-shop-favorites-emerge-from-the-vault-for-a-first-anniversary-encore)

[Product & Features

### Discord Patch Notes: August 30, 2024](/blog/discord-patch-notes-august-30-2024)

[Product & Features

### Discord Update: August 28, 2024 Changelog](/blog/discord-update-august-28-2024-changelog)

[Product & Features

### Queue Up Your Playlists on Discord with the Amazon Music Listening Party Activity!](/blog/amazon-music-activity)

[Product & Features

### Discord Patch Notes: August 1, 2024](/blog/discord-patch-notes-august-1-2024)

[Product & Features

### Now Available: See What’s Happening on Discord, Directly from your Xbox console](/blog/see-whats-happening-on-discord-directly-from-your-xbox)

[Product & Features

### Discord Update: July 26, 2024 Changelog](/blog/discord-update-july-26-2024-changelog)

[Product & Features

### WHO LIVES ON YOUR PROFILE FOR ALL TO SEE? 🎶 SPONGEBOB, IN THE SHOP!](/blog/spongebob-shop-collection)

[Product & Features

### Discord Patch Notes: July 1, 2024](/blog/discord-patch-notes-july-1-2024)

[Product & Features

### Discord Update: June 20, 2024 Changelog](/blog/discord-update-june-20-2024-changelog)

[Product & Features

### How to Join Discord Calls Directly From Your PS5® — No Phone Needed!](/blog/join-discord-calls-directly-from-ps5-no-phone-needed)

[Product & Features

### Feast Your Monit-eyes on Today's Exciting Developer Updates!](/blog/feast-your-moniteyes-on-todays-exciting-developer-updates)

[Product & Features

### Discord Patch Notes: May 2024](/blog/discord-patch-notes-may-2024)

[Product & Features

### Refining Discord’s Mobile Experience With Your Feedback](/blog/refining-discords-mobile-experience-with-your-feedback)

[Product & Features

### Discord Update: May 13, 2024 Changelog](/blog/discord-update-may-13-2024-changelog)

[Product & Features

### Discord Patch Notes: April 2024](/blog/discord-patch-notes-april-2024)

[Product & Features

### Discord Update: April 3, 2024 Changelog](/blog/discord-update-april-3-2024-changelog)

[Product & Features

### Lock in. Stand out. VALORANT arrives in the Shop.](/blog/valorant-shop-collection)

[Product & Features

### Discord Update: March 5, 2024 Changelog](/blog/discord-update-march-5-2024-changelog)

[Product & Features

### Discord Update: December 13, 2023 Changelog](/blog/discord-update-december-13-2023-changelog)

[Product & Features

### Improving Our Mobile Experience](/blog/improving-our-mobile-experience)

[Product & Features

### Discord Update: October 19, 2023 Changelog](/blog/discord-update-october-19-2023-changelog)

[Product & Features

### Avatar Decorations & Profile Effects: Collect and Keep the Newest Styles](/blog/avatar-decorations-collect-and-keep-the-newest-styles)

[Product & Features

### Discord Update: September 13, 2023 Changelog](/blog/discord-update-september-13-2023-changelog)

[Product & Features

### Now Available: Stream Your Xbox Games Directly to Discord](/blog/xbox-stream-to-discord-announcement)

[Product & Features

### Discord Update: July 29, 2023 Changelog](/blog/discord-update-july-29-2023-changelog)

[Product & Features

### Meme Up Some Fun with Remix](/blog/meme-up-some-fun-with-remix)

[Product & Features

### Discord Update: June 22, 2023 Changelog](/blog/discord-update-june-22-2023-changelog)

[Product & Features

### Server Subscriptions Just Got Super Powered: Introducing Media Channels, Tier Templates and more!](/blog/server-subscriptions-updates-media-channels-tier-templates-and-more)

[Product & Features

### Discord Update: May 22, 2023 Changelog](/blog/discord-update-may-22-2023-changelog)

[Product & Features

### Evolving Usernames on Discord](/blog/usernames)

[Product & Features

### Discord Update: April 14, 2023 Changelog](/blog/discord-update-april-14-2023-changelog)

[Product & Features

### Welcome Your New Members Easily with Community Onboarding](/blog/community-onboarding-welcome-your-new-members)

[Product & Features

### Introducing Discord Voice Messages](/blog/discord-voice-messages)

[Product & Features

### April Showers Bring Super-Cool Nitro Powers](/blog/april-showers-bring-super-cool-nitro-powers)

[Product & Features

### New to Discord Nitro: Super Reactions Make Your Emoji Burst to Life](/blog/super-reactions-make-emoji-burst-to-life-discord-nitro)

[Product & Features

### Ready Your Airhorns! 🎺 Discord Soundboard is Coming Your Way](/blog/ready-your-airhorns-discord-soundboard-is-coming)

[Product & Features

### Discord Update: March 20, 2023 Changelog](/blog/discord-update-march-20-2023-changelog)

[Product & Features

### Discord Activities: Play Games and Watch Together](/blog/server-activities-games-voice-watch-together)

[Product & Features

### Discord is Your Place for AI with Friends](/blog/ai-on-discord-your-place-for-ai-with-friends)

[Product & Features

### Now Available: Use Discord Voice Chat on Your PlayStation®5 Console](/blog/playstation-5-voice-integration-announcement)

[Product & Features

### Discord Update: February 20, 2023 Changelog](/blog/discord-update-february-20-2023-changelog)

[Product & Features

### Introducing Video, Screen Share, and Text Chat Support for Stage Channels](/blog/introducing-video-screen-share-text-chat-support-for-stage-channels)

[Product & Features

### Discord Update: January 25, 2023 Changelog](/blog/discord-update-january-25-2023-changelog)

[Product & Features

### Make Your Connection: Connected Accounts Get a Huge Functionality Boost](/blog/connected-accounts-functionality-boost-linked-roles)

[Product & Features

### Announcing Server Subscriptions and the Creator Portal, Now Open to More Communities](/blog/server-and-creator-subscriptions)

[Product & Features

### Discord Update: November 1, 2022 Changelog](/blog/discord-update-november-1-2022-changelog)

[Product & Features

### Attention Server Owners: The App Directory is Here!](/blog/app-directory-is-here-mods-and-admins)

[Product & Features

### Introducing Discord Nitro Basic](/blog/introducing-discord-nitro-basic)

.

[Measure Less to Learn More: Using Fewer, Higher-quality Metrics to Capture What Matters](#)[Multiple Comparisons Problem](#2)[Seeing for Ourselves](#3)[Choosing Metrics for Removal](#4)[Moving Forward](#5)[In Conclusion](#6)

Search

Language

English

Social

Menu

### Product

Product

### Company

Company

### Resources

Resources

### Policies

Policies

Social
