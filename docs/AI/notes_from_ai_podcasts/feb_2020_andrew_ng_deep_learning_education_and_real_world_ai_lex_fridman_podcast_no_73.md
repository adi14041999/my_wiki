# Feb 2020. [Andrew Ng- Deep Learning, Education, and Real-World AI | Lex Fridman Podcast no. 73](https://www.youtube.com/watch?v=0jspaMLxBig)
20 Feb 2020.

## Early days of online education

Teaching machine learning to about 400 Stanford students a year, Ng noticed he was filming the same lecture, telling the same jokes, in the same room every year. Recording it once freed him to spend the saved time building deeper relationships with students instead. That impulse, generalized, became the first MOOCs and later Coursera.

Many of those early videos were shot between 10 p.m. and 3 a.m.— writing the code for the platform and the lecture content in parallel, with a hundred thousand people already signed up and waiting. What kept him going back to the recording studio on a Friday night was, in his words, the principle his whole team held to from day one: **the number one priority is what's best for learners.**

## How big the audience for AI turned out to be

Teaching that class publicly revealed something the field hadn't fully appreciated: the number of people interested in machine learning was far larger than a handful of academic communities. It was developers everywhere, across every country. Ng thinks that growth is nowhere near finished.

He extends the idea further with a literacy analogy. Society once decided that a small class of scribes and authors wasn't enough, and taught (almost) everyone to read and write, which enabled one-to-one communication at scale. Computing, he argues, is still mostly in the "read by monks" phase, where a small population writes code for a large audience. He thinks data science may turn out to be an *easier* entry point into that world than software engineering proper: the owner of a mom-and-pop store may not have an obvious reason to learn to code, but a very simple ability to analyze their own sales and customer data is often immediately useful to them.

## The messiness of real-world data

Software engineering has mature tooling for versioning code. Data, by contrast, is still handled immaturely. He illustrates with a manufacturing example from Landing AI's work in visual inspection. Different human inspectors disagree about whether a scratch on a plastic part is a defect, and the same inspector can disagree with their own morning judgment by the afternoon. At consumer-internet scale with a billion users, you can average the noise away. With 100 examples of a defect, a 10% labeling error rate is a real problem, and dealing with it requires investing directly in the labeling process itself (what the labels mean and how disagreements get resolved) rather than only in the model.

## deeplearning.ai: getting started

Of Ng's three current ventures, [deeplearning.ai](https://www.deeplearning.ai) is the education arm. Courses like the Deep Learning Specialization that teaches the mechanics of neural networks, CNNs, RNNs and attention from first principles.

A theme Ng returns to is that the specialization tries to teach the *practical* know-how of making these systems work, not just the theory: recognizing when a model is overfitting, when collecting more data will help versus wasting six months. He draws a direct analogy to software debugging: people who are good at debugging machine learning systems are, in his estimate, 10 to 100 times faster at getting something to work than people who aren't.

## Unsupervised and self-supervised learning

Asked for the most beautiful idea in deep learning, Ng picks self-supervised learning. His running example: take an unlabeled image, rotate it by a random multiple of 90 degrees, and train a network to predict the rotation actually applied. The task itself is free (any image gives you infinite labeled examples), but the representation the network learns along the way transfers well to other tasks.

He lists a few more instances of the same trick: predicting a masked-out word in a sentence, and the "jigsaw" task of cutting an image into a 3×3 grid, shuffling the pieces, and having a network predict which of the 9! permutations produced the shuffle. See the wiki's [self-supervised learning](../deep_learning_for_computer_vision/self_supervised_learning.md) page for more on this.

## Study habits

Ng's advice for learning deep learning centers on regularity over intensity. He reads or studies every Saturday and Sunday for the same reason— removing the daily choice removes the friction.

He also takes handwritten notes even in videotaped courses, because handwriting is slow enough that it forces you to recode what you heard into your own words, and that recoding is what drives long-term retention. Typing verbatim, which is fast enough not to require rephrasing, retains less.

## A career in deep learning

Ng's advice for building a career: start with coursework, because it's the most time-efficient way to master material that's already well understood, even if it means being less immediately productive in the first couple of years. Once you've exhausted what courses can efficiently teach, the path is projects, then papers and blogs, and doing something small rather than waiting for a project big enough to feel meaningful.

On whether to pursue a PhD: Ng doesn't think it's required to have a large impact in this field, but there are several good paths (a PhD at a top program, a job at a top AI team, starting a company), and a PhD is close to necessary only if the specific goal is an academic professorship. His stronger piece of advice concerns something people underweight: **who your day-to-day peers and manager are matters more than the company's or university's name.**

## AI Fund: a startup studio

AI Fund grew out of Ng's time leading Baidu's AI group, where part of his job was to systematically spin up new lines of business from the company's AI capabilities. The self-driving car team and a smart-speaker product (which shipped before Amazon's) both originated this way. He found that generative role more fun than running an existing business unit, and built AI Fund as a "startup studio": an organization that manufactures new companies from scratch, rather than only investing in ones that already exist.

His observed failure mode for startups is building something technically impressive that no one asked for. His corrective is being relentlessly customer-obsessed, since in the long run only the customer decides whether a company survives. He's also selective about which ideas to pursue even when they'd be commercially viable. For example, he describes killing a viable idea for a video-recommendation company because its actual effect would have been to get people to watch more video with no redeeming value, which didn't meet his own bar for making the world better.

## Landing AI: bringing AI to established industries

Landing AI works the other side of the same problem: helping already-established companies, mostly outside software and internet, adopt AI.

His concrete example is visual inspection in manufacturing: a camera and a model replacing (or assisting) a human checking parts for defects. The practical failure mode there is different from what shows up in papers— data sets are small, and factory conditions drift constantly in ways a fixed test set doesn't capture (lighting changes, and in one real anecdote, a bird got into a factory and left droppings on equipment that changed the visual environment enough to matter).

His advice for a company's first step is the same lesson from his own Google days: **start small.** His first internal customer at Google Brain wasn't search or ads. It was the Google speech team  helping ship a more accurate speech recognizer, then Google Maps reading house numbers from Street View to improve address accuracy. Only after two visible wins did a conversation with the ads team become possible. Early small deployments also teach the team itself how the technology behaves in practice.

## AGI, alignment, and what he worries about instead

Ng is less interested in the standard existential framing of the alignment problem.

His concrete near-term concerns: **bias** in deployed systems, **wealth and power concentration** (AI and the internet enabling winner-take-most dynamics across industries beyond just tech, the way ride-sharing apps reshaped the taxi industry), and the adversarial use of tools like deepfakes.

## Closing thoughts

He ends, fittingly, with his own maxim: ask whether what you're working on, if it succeeds beyond your wildest dreams, would have significantly helped other people— and if not, keep searching.
