# Apr 2019. [Ian Goodfellow- Generative Adversarial Networks (GANs) | Lex Fridman Podcast no. 19](https://www.youtube.com/watch?v=Z6rxFNMGdn0&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
18 Apr 2019.

## The limits of deep learning

The *Deep Learning* book opens with a Russian-doll diagram: deep learning inside representation learning, inside machine learning, inside AI. Lex uses it to ask where the limits of DL are.

Goodfellow's first answer is **data**. Deep learning needs a great deal of it, especially labelled data. Unsupervised and semi-supervised methods reduce the labelling requirement but still want a lot of unlabelled examples. Reinforcement learning needs no labels but needs enormous quantities of *experience*. Improving **generalization** is the single most important bottleneck.

His second answer is about scope. Nobody proposes deep learning as the *entire* ingredient of intelligence. It is used as a submodule of larger systems. AlphaGo has a deep learning model estimating the value function, most RL algorithms have one estimating which action to take next. Deep learning is essentially a **function estimator** dropped into a bigger architecture.

## Neural networks as programs

Asked whether neural networks could be made to reason the way symbolic systems did (producing something more like programs than functions), Goodfellow says we already see a little of this, because he thinks of neural nets as a kind of program in the first place.

The framing is geometric. Draw the computation graph of a model:

- The **depth** is the number of steps that run in sequence.
- The **width** is the number of steps that run in parallel.

Shallow learning, back when he started, meant things like support vector machines: many input features, each multiplied by a weight, but all those multiplications happening *in parallel*. Little was done in series. What deep learning really bought us, he argues, is the ability to have **steps of a program that run in sequence**.

When deep learning took off academically in 2006, with Hinton training deep belief networks, everyone thought each layer learned a different level of abstraction: edges, then corners, eventually grandmother cells recognizing specific objects. Goodfellow says most people now think of it as a computer program, where more layers means more updates before you output the final number. **Nobody believes layer 150 of a ResNet is a grandmother cell and layer 100 is contours.** ResNets take one particular kind of representation and update it repeatedly.

So the network is not replacing its representation at each step. It is **refining** it. And that, he suggests, is a little like reasoning. Not reasoning in the form of deduction, but taking a thought and refining it carefully until it is good enough to use.

## Writing the deep learning chapter for AIMA

Goodfellow wrote the deep learning chapter for the fourth edition of Russell and Norvig's *Artificial Intelligence: A Modern Approach* (see the [Peter Norvig episode](sep_2019_peter_norvig_artificial_intelligence_a_modern_approach_lex_fridman_podcast_no_42.md), where Norvig describes the same collaboration from the other side).

He waited about a year before writing anything. Even having written a full-length textbook, covering everything in one chapter is intimidating. What helped was **watching how the field changed after his own book came out**. Seeing which topics turned out to be extraneous, and which survived a few years of publication. It also helped that the field had stabilized to the point where some core 1980s ideas are used again; when he started studying machine learning, almost everything from the 1980s had been rejected, and some of it has since come back. The material that stood the test of time is what he focused on.

He also frames a choice between two philosophies of book-writing:

1. A **reference** that covers everything.
2. A **high-level summary** giving readers the language to understand a field and telling them which concepts matter most.

*Deep Learning* with Bengio and Courville sat somewhere between the two. The AIMA chapter let him focus on the second. In several places he simply wrote paragraphs saying *here is a rapidly evolving area you should pay attention to*. It is pointless to describe the latest learn-to-learn model, because you should know that learning to learn is a thing and may well produce the best module for your next project, without needing a summary of which architecture currently holds which score.

## What is deep learning?

Goodfellow's own definition: **any machine learning that involves learning the parameters of more than one consecutive step.**

Shallow learning is where you learn many operations that happen in parallel. You might have multiple steps (hand-designed feature extractors, say), but only one step is *learned*. Deep learning is anything with multiple learned operations in sequence. That includes convolutional and recurrent networks, and it also includes things that have died out, such as Boltzmann machines, where backpropagation was not used.

He notes that many people today define deep learning as *gradient descent applied to differentiable functions*, and calls that a legitimate usage— just not his. The disagreement becomes clear through his three-part decomposition of any machine learning algorithm:

1. **The model**— how you take data and parameters and produce a prediction. A neural net, a Boltzmann machine, a recurrent model.
2. **The optimization algorithm**— how you update the parameters or the state.
3. **The dataset**— how you represent the world as it enters the system.

"Deep" is a statement about **the model**: it has multiple separately parameterized steps, whether those are layers of a feed-forward differentiable computation or layers in a graphical model. Gradient descent is a statement about **the optimizer**. So training a convolutional net with evolution or a genetic algorithm would still, in his usage, be deep learning. And Boltzmann machines are the main example of a model where you cannot take a derivative for learning, yet which clearly applies many steps of processing during inference.

### Beyond backpropagation

On Hinton's suggestion that we throw away backprop and start over, Goodfellow does not expect it to disappear entirely. His general observation about the field:

> Most of this time when we decide that a machine learning algorithm isn't on the critical path to research for improving AI, the algorithm doesn't die, it just becomes used for some specialized set of things.

Logistic regression is the example. Unexciting to anyone working on speech or autonomous cars, yet still heavily used for noisy data in medicine and finance, and for rapid predictions under tight time limits. Backprop and gradient descent are around to stay, but may not be everything needed for human-level AI.

He is optimistic that something better will be found. One shape it might take: **stacks of models where lower-level models predict the parameters of higher-level ones**, so that at the top you are not literally computing gradients but predicting how different values will perform. This already exists in miniature in Bayesian optimization, where a Gaussian process predicts how well different parameter values will do, and is used for hyperparameter search. We know many non-backprop methods that work well on specific problems; what we have not found is a way to take one of them and have it genuinely advance the state of the art on an AI-level problem.

## What GANs are

A **generative model** trains on a set of data (a collection of cat photos) and either generates more of it or estimates a probability distribution over it, so you can ask how likely a new image is to be a cat. Some generative models are good at creating new data; others are good at estimating that density function. GANs are focused on **generating samples** rather than estimating density.

What makes GANs specific is that they set up a **two-player game** in the game-theoretic sense:

- The **generator** produces output data such as images. At the start of training it produces completely random images.
- The **discriminator** takes images as input and guesses whether they are real or fake. It is trained on real photos from the training set, labelled real, and on images from the generator, labelled fake.

As the two compete, the discriminator gets better at telling real from fake and the generator gets better at fooling it. Analyzed through game theory, there is a **Nash equilibrium** at which the generator has captured the correct probability distribution (perfectly realistic cat photos) and the discriminator can do no better than random guessing, because samples from both sources look equally likely to have come from either.

!!! note "The mathematics"
    The wiki's [Generative Adversarial Networks](../deep_generative_models/generative_adversarial_networks.md) page covers the formal version of this— the [minimax objective](../deep_generative_models/generative_adversarial_networks.md#the-minimax-game), the [optimal discriminator](../deep_generative_models/generative_adversarial_networks.md#optimal-discriminator), the [connection to Jensen–Shannon divergence](../deep_generative_models/generative_adversarial_networks.md#connection-to-jensen-shannon-divergence), and the [training challenges](../deep_generative_models/generative_adversarial_networks.md#challenges).

## Why is it surprising that this works at all?

You can take a convolutional network, not learn its parameters at all, and the architecture alone is already useful for tasks like inpainting. That suggests the convolutional architecture captures something important about the structure of images, and that learning is not required to extract all of it.

The implication is uncomfortable. If much of the success rests on architecture that reverse-engineers the human visual system, it would be **much harder to build generative models in other domains**. Speech models work reasonably, but the field has not explored many different kinds of data. For example, there are few deep generative models of, say, biology datasets measuring enzyme levels. The same trick may not transfer to arbitrary data.

## A taxonomy of generative models

Most generative models are **likelihood-based**: the model tells you how much probability it assigns to an example, and you maximize the probability assigned to the training data.

The difficulty is computational. It is hard to design a model that can create really complicated images or audio *and* whose likelihood function you can actually evaluate. For most models you would write down intuitively, calculating the probability assigned to a particular point is nearly impossible.

One school within the likelihood family carefully designs the model so that measuring density stays tractable. [Autoregressive models](../deep_generative_models/autoregressive_models.md) do this by factoring the distribution into a product over every feature— estimating the probability of each pixel given all the pixels before it. There are tricks to compute the density for all pixels roughly in parallel, but **generating** an image still tends to require going one pixel at a time, which is slow. Hierarchical variants keep the runtime under control, and the image quality, runtime aside, is reasonable.

Goodfellow thinks the best results these days come from GANs, but adds a genuinely candid caveat about how hard it is to attribute that.

## Semi-supervised learning

You can use GANs to learn classifiers without having labels for every example.

Tim Salimans' paper *Improved Techniques for Training GANs* (Goodfellow is a co-author but disclaims credit for this part) showed you can take the GAN **discriminator and use it as a classifier**— not merely saying real or fake, but saying specifically which kind of object a real image contains.

The numbers are striking. Getting below 1% error on MNIST required around **60,000 labelled examples** up to about 2014. In 2016 the semi-supervised GAN got below 1% using only **100 labelled examples**— roughly a 600× reduction in labels. It still uses many images; they simply do not each need a label.

The follow-on question is generating recognizable objects of a particular class, which still seems to need labels. A group at Brain Zurich, building on DeepMind's BigGAN, showed they could match its performance using about **10% of the labels**. BigGAN was trained on ImageNet— roughly 1.2 million images, all labelled.

Their mechanism is a **clustering algorithm**: the discriminator learns to assign objects to groups, and knowing that objects fall into archetypal groups helps the model form more realistic ideas of what should appear in an image, since every image it creates has to come from one of them.

The failure mode without labels is instructive. An unlabelled GAN tends to produce things that look like grass, water, brick or dirt— plausible textures, without much going on. Goodfellow's explanation is that in a large ImageNet image the object does not occupy the whole frame, so the model learns to create realistic *sets of pixels* without learning that **the object is the star of the show** and should be in every image.

Lex raises the related CycleGAN observation: horses are usually photographed on grass and zebras on drier terrain, so translating between them drags the background along— you end up generating zebras on grasses. The model has not learned to segment the object from its setting.

## Other adversarial games

**Security:** Most interactions can be modelled as a game between an attacker trying to break your system and a defender trying to build a resilient one. This is where Goodfellow spends most of his time.

**Domain adversarial learning:** An approach to domain adaptation that looks a great deal like GANs.

Domain adaptation is training a model in one setting and deploying it in another, wanting good performance despite the difference. You might train on a clean dataset like ImageNet and deploy on users' phones, where pictures are taken in the dark, while moving, and are not well composed. A normal model often degrades badly.

The adversarial version trains a **feature extractor** whose features have the same statistics regardless of domain. The two players are:

- The **domain recognizer**, which looks at the extracted features and guesses which domain they came from— closely analogous to the real-versus-fake discriminator.
- The **feature extractor**, loosely analogous to the generator, which tries both to fool the domain recognizer *and* to extract features that are good for classification.

When it works, you get features that behave about the same in both domains.

## Data augmentation

The simplest hope— train a GAN on a limited training set, generate more data, train a classifier on the enlarged set, and get better test performance— is something Goodfellow has **never heard of working**.

His reasoning about why is clean. The hope requires the GAN to generalize better than the classifier would have, trained on the same data, and there is no reason to believe that. But there is a weaker thing you might hope for: that the GAN generalizes **differently**.

That suggests an experiment he has not tried and thinks someone should: train many different generative models on the same training set, sample from all of them, and train a classifier on the combined output. Each generative model might generalize in a slightly different way and capture different axes of variation, and the classifier could absorb all of them.

## Fairness

The core difficulty: leaving a sensitive variable out of the input is not enough, because it can often be inferred from other attributes. Goodfellow's own example is that if you have someone's name but not their gender, and the name is Ian, the gender is fairly obvious.

What you want is a model that can take in many attributes and make an accurate, informed prediction while you remain confident it is not **reverse-engineering the sensitive variable internally**. The construction is the domain-adversarial one again: a feature extractor playing against a feature analyzer, where you ensure the analyzer cannot guess the value of the protected variable from the features.

A second, more speculative idea uses CycleGAN-style translation for **auditing**. We have seen horses turned into zebras, and Ming-Yu Liu's unsupervised models turning day photos into night photos. For fairness you could imagine taking records of people in one group, converting them into analogous people in another group, and testing whether the system treats them equitably. Goodfellow is careful that a great deal would have to be right— above all, making the conversion process itself fair— and says it is nowhere near usable yet.

## Are there still quick breakthroughs?

Goodfellow thinks many ideas can still be developed quickly.

What has changed is the **cost of proving an idea works**. Having the GAN idea today would be much harder to validate than in 2014, because you would need results on something like ImageNet or CelebA at high resolution, and those take a long time to train. In 2014, MNIST was enough.

The areas he thinks are ripe for low-resource, high-payoff work are **fairness and interpretability**, because we do not yet know how anything there should be done. For interpretability we do not even have the right definitions— everyone has a different idea in their head, and what gets discussed is essentially opinion.

His model for what progress looks like is **differential privacy**. Cynthia Dwork and her collaborators produced a technical definition of privacy where things had previously been mushy. With that definition you could design randomized algorithms for accessing databases and *guarantee* individual privacy in a quantitative sense. Once the definition existed, the algorithms came quickly.

So defining a measurable concept related to interpretability could have a huge impact on the field, and the algorithms providing guarantees on that quantity would likely follow fast.

## What would count as progress toward general intelligence

On what it takes: **better environments** for training agents, giving them a wide diversity of experiences, and a great deal of computation.

He thinks simulation is a necessary ingredient, and does not think we get to general intelligence by training on fixed datasets or by thinking hard about the problem. The agent needs to interact and to accumulate a variety of experiences **within the same lifespan**.

The gap he identifies is integration. Today we have many models that each do one thing, trained on one dataset or one RL environment. There are papers on getting one set of parameters to perform well across many RL environments, but those environments tend to be similar (for example, all action-based video games). We have nothing that goes **seamlessly from one type of experience to another** and integrates everything it does over a lifetime: from playing a video game, to reading the Wall Street Journal, to predicting how effective a molecule will be as a drug.

### His test of intelligence

Asked what would genuinely impress him, Goodfellow gives an unusually concrete answer: **something that accomplishes a task without a lot of glue from human engineers.**

Instead of going to the CIFAR-10 website, downloading the data, writing a Python script to parse it and so on, you would **point an agent at the CIFAR-10 problem**, and it downloads the data, extracts it, trains a model, and starts giving you predictions. You might give it the URL for Wikipedia, or type a paragraph explaining what you want, and it works out which web searches to run.

> If something knows how to pre-process the data so that it successfully accomplishes the task, then it would be very hard to argue that it doesn't truly understand the task in some fundamental sense.

He does not claim this is the philosophical definition of intelligence— only that it would be genuinely useful, would impress him, and would convince him of a real step forward.
