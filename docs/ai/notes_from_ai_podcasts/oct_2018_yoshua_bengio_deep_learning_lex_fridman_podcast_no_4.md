# Oct 2018. [Yoshua Bengio- Deep Learning | Lex Fridman Podcast no. 4](https://www.youtube.com/watch?v=azOmzumh0vQ)
20 Oct 2018.

### Credit assignment

Biological Neural Networks can do ‘credit assignment’ over a long period of time. **The credit assignment problem in neural networks, especially in the context of reinforcement learning, refers to the challenge of determining which actions or decisions are responsible for a given outcome, particularly when there's a delay between actions and feedback.** This makes it difficult for the network to learn effective policies.

Why is it a problem? **If the network can't accurately determine which actions are responsible for good or bad outcomes, it will struggle to learn effective strategies.** 

Understanding how the brain solves the credit assignment problem is a key area of research in neuroscience and artificial intelligence.

We store all kinds of memories in our brain which we can access later in order to help us- 

- infer causes of things that we are observing now
- assign credit (determining which actions or decisions are responsible for a given outcome) to decisions or interpretations we came up with a while ago when those memories were stored.
- Furthermore, we can change (update) the way we would have reacted or interpreted things in the past to new scenarios to attempt to achieve good outcomes (in simple words, learning from mistakes). That’s credit assignment used for learning.

Humans seem to be able to do credit assignment through essentially arbitrary times (we could remember something we did last year and then now because we see some new evidence we can change our minds about the way we were thinking last year and hopefully not do the same mistake again). Part of the reason for that is probably forgetting. You're only remembering the really important things it's very efficient. 

### Current state of deep learning

**Instead of learning separately from images and videos on one hand and from text on the other hand we need to do a better job of jointly learning about language and about the world to which it refers.** This way both sides can help each other. We need to have good world models in our neural nets for them to really understand sentences which talk about what's going on in the world. **We need language input to help provide clues about what high-level concepts like semantic concepts should be represented at the most processed levels of these neural nets.**

### Training objectives and frameworks

The training objectives-

- **which could be important to allow the highest level explanations to rise from from the learning**
- **which could be used to reward exploration (the right kind of exploration)**

**and the training frameworks (for example, going from passive observation of data to more active agents which learn by intervening in the world the relationships between causes and effects)** are neither in the dataset nor in the architecture. These are more crucial to take us closer to AGI.

### Learning through interaction

Children learn by interacting with objects in the world- an idea largely absent from artificial neural networks except in some reinforcement-learning settings. One can imagine an objective rewarding an agent for interactions (such as poking an object in a certain way) that help it learn further. Evidence from infants supports this. They are not passive learners but direct their attention toward the aspects of the world that are most interesting and surprising in a non-trivial way. Due to this process, they revise their theories of the world.

Even state-of-the-art deep learning fails to learn good models of very simple environments (such as small grid worlds). Where a human needs dozens of examples, these methods need millions, even for trivial tasks. This is an opportunity for academics without massive compute to do important work on training frameworks and agent learning in simple, synthetic environments.

There's an opportunity for academics who don't have the kind of computing power to do really important and exciting research to advance the state-of-the-art in training frameworks and agent learning in even simple environments that seem trivial but yet current machine learning fails on.

### Knowledge

In the 1980s, AI focused on knowledge representation, knowledge acquisition, and expert system. This is called the symbolic AI way of representing knowledge (using discrete, human-readable symbols and explicit rules that manipulate them, rather than the continuous numerical vectors a neural net uses). That approach was largely put on hold because it did not work, but its goals remain important. One reason expert systems failed is that much of our knowledge, common sense and intuition can't be introsepcted or put into words. We make many decisions we cannot really explain. Such knowledge is necessary for good decisions yet hard to codify in rule-based formalisms.

There is something powerful about **distributed representations**, which is what makes neural nets work so well, and it is hard to replicate in a symbolic world. But there is a trade-off: knowledge in expert systems is neatly decomposed into rules, whereas a neural net is a big blob of parameters that work intensely together to represent everything the network knows. But the weakness of this form of representation is that it can't be sufficiently factorized. This is one of the weaknesses of current neural nets that we have to take lessons from classical AI in order to bring in another kind of compositionality which is common in language.

Beyond separating the high-level variables (the meaningful factors of variation, e.g. an object's identity, position, color, or pose), we must also disentangle the *mechanisms* (the "rules") that relate them, so each piece of knowledge lives on its own. Otherwise networks suffer **catastrophic forgetting**: learning new things destroys what was learned before. Better-factorized knowledge avoids much of this.

There's the sensory space like pixels where everything is tangled. The information like the variables are completely interdependent in very complicated ways. So is the computation. We can hypothesize a right high-level representation space where both the variables and how they relate to each other can be disentangled and that will provide a lot of generalization power.

### Generalizing to new distributions

Current machine learning typically assumes the test distribution matches the training distribution. In our training methods, distribution of the test set is similar to the distribution of the training set. This is where current machine learning is too weak. It doesn't tell us anything about how to generalize a new distribution. This is a key weakness: it tells us nothing about how to generalize to a new distribution. Yet humans generalize to new distributions all the time, because different distributions still have things in common. For example, a science-fiction novel may take place on another planet that looks very different on the surface but obeys the same laws of physics. We understand the story because we transport our knowledge from Earth (about underlying cause-and-effect relationships, physical mechanisms, and even social interactions). So we make sense of a world that is visually completely different.

### Bias in machine learning

What can be done about bias and ethics in machine learning?

In the short term, techniques already exist (and will keep improving) to measure bias in datasets and build less-biased classifiers mature enough that regulators could require their use despite a small accuracy cost. 

In the long term, the harder goal is instilling **moral values** into computers. There is work already on detecting emotions in images, sounds, and text, and patterns such as injustice that trigger anger. 

### Machine Teaching

Supervised learning has had a lot of success. The broader problem is Machine Teaching. What are good strategies for teaching a learning agent, and can we train a system to be a good teacher? In one project (the "BabyAI" game), a learning agent and a teaching agent interact; the teacher uses its knowledge of the environment to help the learner learn as quickly as possible.

### The Turing Test

The Turing Test, originally called the **"Imitation Game"** by mathematician Alan Turing in 1950, is a benchmark for artificial intelligence. It measures a machine's ability to exhibit conversational behavior indistinguishable from that of a human.

The standard evaluation involves three participants in separate locations.

1. The Interrogator (Judge): A human who asks questions through a text-based interface.
2. Player A: A human respondent.
3. Player B: An AI system attempting to deceive the interrogator.

The interrogator evaluates the transcripts of the conversations without knowing which participant is human and which is the machine. If the interrogator cannot reliably tell the difference, the machine is said to have passed the test.

The hardest part of the conversation for machines is everything involving **non-linguistic knowledge**. For example, cases such as **Winograd schemas** (sentences that are semantically ambiguous unless you understand enough about the world). 

These point toward building systems that understand how the world works and its causal relationships. Passing the Turing Test should be largely independent of language; differences between languages are minute in the grand scheme.