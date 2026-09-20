# Aug 2019. [Jeremy Howard- fast.ai Deep Learning Courses and Research | Lex Fridman Podcast no. 35](https://www.youtube.com/watch?v=J6XcP4JOHmk&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
27 Aug 2019.

## Programming environments he has loved

**Microsoft Access**, of all things, is his answer for the best programming *environment* he has used— Visual Basic for Applications, which he is clear is not a good language. Access let you build tables, relationships and queries graphically, tie them to forms, set up event handlers and calculations, and produce reports. Not for massive scalable systems, but for useful small applications.

His observation about why this matters is that he has always connected programming languages to **the ease of managing data**: his interest throughout has been getting some data, doing something with it, and putting it back out. Programming on top of a relational database today is more of a headache than it should be— you need a database server (or SQLite, with its own issues), then an ORM for a decent programming model, then all the pieces tied together.

**Delphi** is the language he most loved after that. Delphi was a compiled, fast language that was as easy to use as Visual Basic.

## APL, J and the array languages

Asked which language he actually likes, Howard names **J**, and to explain it he sketches a piece of history most people do not know.

There were roughly three major directions in programming language design:

1. The **lambda calculus / Alonzo Church** direction— Lisp and its descendants.
2. The **imperative and object-oriented** direction— Algol, Simula, C, C++.
3. The **array-oriented** direction, which began with Ken Iverson.

The third started not as a programming paper but as a mathematics paper, on **notation as a tool of thought**. Iverson's idea was a new kind of mathematical notation that would be more flexible, more expressive and better defined than traditional mathematical notation. Iverson then turned the notation into a language and, since the name was available, called it *A Programming Language*: **APL**.

Iverson and his son later built **J** on top of everything learned from APL, and Howard calls it the most expressive, composable, beautifully designed language he has seen. It is array-oriented, which means **you generally do not write loops**— everything is done with an extreme version of broadcasting, familiar in miniature from NumPy. You do a great deal in one line, it looks like mathematical notation, and because a single screen of code is usually enough for a whole program, you can keep it all in your head and communicate it clearly.

APL produced two main branches. J is an open-source niche community of enthusiasts. **K** went the other way: an extraordinarily expensive language used by some of the richest hedge funds, where the entire machine is small enough to sit inside L3 cache and easily wins every data-processing benchmark he has seen. You rarely encounter it because it costs on the order of $100,000 per CPU.

His verdict is that these languages are far more powerful in every way than what almost everyone uses daily— with the caveat that they are focused on computation, and little work has gone into user interfaces or the surrounding ecosystem.

## What he wants from a language for machine learning

Howard hopes Swift succeeds, and the reason is the word Chris Lattner uses: **infinitely hackable**. He wants something where he, his collaborators and his students can look at and change everything top to bottom, with nothing mysterious or inaccessible.

Python is the opposite, because it is slow. You reach a floor where everything below is C. And at that point your debugger stops working the same way, your profiler stops working the same way, your build system stops working the same way.

The point is not raw speed for its own sake; it is **research velocity**. His view is that our understanding of deep learning is primitive and most things do not work very well, so every domain he looks at has obviously stupid things that need fixing. He wants to jump in and experiment quickly.

### Why this is hard, and what might fix it

Howard is careful that nobody is at fault. Writing an acceptably fast GPU program is too complicated regardless of language. You are dealing with 10,000 threads, synchronization, grid blocks, warps: so much boilerplate that you must be a specialist and it may take a year to optimize one algorithm.

The promising direction is compiler work: **Tensor Comprehensions, Tile, TVM**, and above all, [MLIR](https://mlir.llvm.org/), which Lattner leads. The idea is domain-specific languages for tensor computation, with a compiler that optimizes them.

His dream stack is Swift CUDA kernels written concisely in something that looks a bit like J or APL, Swift layers on top, and SwiftUI above that.

He also notes the hardware politics. It all bottoms out in CUDA and NVIDIA today. NVIDIA is massively overcharging for enterprise cards because there is no serious competition, since nobody else does the software properly. And [TPUs](may_2019_chris_lattner_compilers_llvm_swift_tpu_and_ml_accelerators_lex_fridman_podcast_no_21.md#what-a-tpu-is) are worse: Google made an explicit decision to keep them almost entirely unprogrammable, because they felt there was too much IP in there and direct access would reveal their secrets. You cannot program the memory directly or run code you can inspect on the machine. Everything goes through a virtual machine, leaving you plugging high-level pieces together.

## Privacy and doing more with less data

Howard's angle on privacy is unusual: he attacks the premise that you need the data at all.

Most vendors are **strongly incentivized to make you believe you need more data and more computation**. He names Google and IBM as pushing the idea that they have more data, more compute and more intelligent people than anybody else, so you must trust them because nobody else can do it.

What they have found is that you very often do not need much data at all— the data already inside your organization is usually enough for state-of-the-art results, if you know **transfer learning**.

The model he prefers is individual control. He helped a startup, doc.ai, whose app lets you download your medical records from your providers to your phone and upload them again at your discretion, sharing with whom you choose.

## Why fast.ai exists

Before starting it, Howard spent a year researching where the biggest opportunities for deep learning were. He had worked with neural nets for over 20 years and knew from Kaggle that deep learning was rapidly becoming state of the art in every area anyone applied it to— and that theoretically it would do so in nearly every domain.

The frustration was that there were far too many domains to pick from.
So he inverted the approach:

> Rather than me picking an area and trying to become good at it and building something, I should let people who are already domain experts in those areas, and who already have the data, do it themselves.

His background makes this personal— ten years in management consulting at McKinsey, a lot of time with domain experts, and the conviction that domain expertise is where value is generated in society. Most of those people cannot code, and cannot invest three years in a graduate degree. Giving them the tool seemed like the biggest societal impact he could have.

## Theory versus practice

His diagnosis is structural rather than personal. Scientists need to publish, which means working on things their peers already recognize, which means everyone works on the same thing, and nothing encourages work that is practically useful. The result is a great many minor advances in heavily studied areas with no significant practical impact.

Meanwhile the things that would change the world go unstudied:

- **Transfer learning:** Do better at it and suddenly far more people can do world-class work with fewer resources and less data. Almost nobody works on it.
- **Active learning:** How to get more out of the humans in the loop. Almost nobody works on it because it is not trendy.

## Against big compute

Howard thinks multi-GPU and especially multi-machine training is **largely a waste of time**, on the grounds that anything slowing your iteration speed is a waste of time. Use it for the final perfecting of a model if you must.

He extends this to ImageNet itself: why test on 1.3 million images when most people do not have 1.3 million images, and their own research shows smaller subsets give the **same relative answers**? So he released two datasets:

- **Imagenette**— a small, deliberately easy-to-classify subset of ImageNet (spelled with the French ending, and pronounced accordingly).
- **Imagewoof**— a subset containing only dog breeds, which is hard.

Both train on a single GPU in ten minutes, and the results transfer to ImageNet nearly all the time.

His counter-claim: he expects all the major AI breakthroughs of the next twenty years to be doable on a single GPU, and notes that **none of the big breakthroughs of the last twenty years required multiple GPUs**— batch norm, dropout, the original [GANs](apr_2019_ian_goodfellow_generative_adversarial_networks_gans_lex_fridman_podcast_no_19.md).

## Learning rates and super-convergence

This comes from **Leslie Smith**, a researcher who cares about training networks quickly and accurately. He discovered **super-convergence**: certain networks with certain hyperparameter settings can suddenly be trained **ten times faster by using a ten times higher learning rate**.

The paper was not published, and Howard's account of why is the most pointed observation in the episode:

> Deep learning in academia is not considered an experimental science. So unlike in physics, where you could say "I just saw a subatomic particle do something the theory doesn't explain" and publish that without an explanation (and then over the next 60 years people work out how to explain it) we don't allow this in the deep learning world.

Leslie could not publish "I saw this train ten times faster than it should have, I don't know why," because the reviewers said they could not publish it without knowing why. Howard notes he reads more unpublished papers than published ones, because that is where the interesting insights are.

The technique's core trick is **starting at a very low learning rate and gradually increasing it**, taking very small steps at first and eventually much bigger steps than anyone thought possible. There is a double benefit: training ten times faster also generalizes better, because fewer epochs means looking at the data less.

## Working with data

His argument for why this matters is that interpreting a model **makes you a domain expert faster**, because it tells you which features are most important and which groups are being misclassified.

## Teaching

**What teaching taught him.** Two things he wanted to believe but had no strong evidence for, and now does:

1. **Anyone can do it.** There is a lot of snobbishness about who can learn to code or do AI, and he calls it bullshit. He has seen people from many different backgrounds get state-of-the-art results in their own domains.
2. **The differentiator is tenacity.** It seems to be the only thing that matters. Many people give up; of those who do not, pretty much everybody succeeds.

## Advice for getting started, and for getting good

The advice is the same at both timescales: **train lots of models**.

For beginners, the emphasis is on *fine-tuning your own*, not running someone else's, because only then is it a model in your domain. It takes five minutes to fine-tune a model on data you care about, and lesson two teaches you to build your own dataset from scratch by scripting Google image search, clean it with widgets in the notebook, and deploy a web application.

Then: print the inputs, print the outputs, change the inputs, watch how the outputs vary. Run lots of experiments to build intuition.

For becoming an expert over years, his answer is pointed:

Become the expert in your passion area and combine the tool with your domain expertise.

You need a real problem, because otherwise **how do you know whether your results are any good, or why they are bad?**

## Startups

The answer is the same word again: **not giving up**. You fail by running out of money or time, so keep costs low and save beforehand so you can afford the time.

Beyond that: **work on something you understand and care about**. The biggest mistake he sees is people doing a PhD in deep learning and then trying to commercialize the PhD, which is a waste of time because the topic was chosen as an interesting research exercise rather than an actual problem.

He self-funded his first two companies and thinks that was right. **VC-backed startups are much scarier**, because you have people on your back who do this all the time telling you to grow, who do not care if you fail— only if you do not grow fast enough. He also found it hard to do the thing he thought was right for the company rather than the thing that would make the investors happy; they always tell you not to do the latter, and then get upset when you do not.

On exits: a VC exit needs to be a thousand times over, whereas selling something for ten million dollars means you have made it. Both are perfectly good outcomes — it depends whether you want to build something you would be happy to do forever.

## Spaced repetition

Howard uses **Anki** heavily and says it works incredibly well for him.

**Spaced repetition** comes from Hermann Ebbinghaus, who did something tedious about 150 years ago: he wrote down random sequences of letters on cards and tested how well he remembered them a day or a week later. He found a forgetting curve— the probability drops sharply the next day, then a little more each day. Crucially that **revising a card flattens the curve**. From that he worked out a roughly optimal revision schedule.

A program like Anki does this for you, and rescheduling on failure makes it, by definition, a way of being guaranteed to learn something.

## What he worries about

On the next breakthrough, he declines to predict, and thinks we do not need one:

> What we already have is an incredibly powerful platform to solve lots of societally important problems that are currently unsolved.

On human-level intelligence and when it arrives, he is dismissive of the question itself— there is no data and nothing to go on, and there are too many societally important problems right now for him to find it interesting.

What he does worry about:

**Labour force displacement.** He calls the econometric argument (that previous technologies did not cause mass displacement, so this one will not) frivolous, and says he is desperately concerned. The changing workplace has already hollowed out the middle class, and students leaving school today face a less rosy financial future than their parents did, which has not happened in centuries. He sees it turning into anxiety, despair and violence.

**Ethics, as a professional obligation.** His argument is that every data scientist working with deep learning is using an extremely high-leverage tool, and researchers should recognize their work will be used by practitioners. The questions he thinks they must own:

- How will humans be in the loop here?
- How do we avoid runaway feedback loops?
- How do we ensure an appeals process for people affected by the algorithm?
- How do we ensure the constraints of the algorithm are explained ethically to the people using it?

These are human issues that **only data scientists are positioned to educate people about**. Data scientists tend to think of themselves as engineers who need not be part of that process, which he says is simply wrong.
