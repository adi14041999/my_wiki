# Nov 2018. [Vladimir Vapnik- Statistical Learning | Lex Fridman Podcast no. 5](https://www.youtube.com/watch?v=STFcvzoxVw4&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
16 Nov 2018.

## Instrumentalism and realism

Asked whether God plays dice, Vapnik answers that it only looks that way because there are factors we do not know. He then reframes the question through a distinction from philosophy of science:

- **Instrumentalism**— You create theories in order to *predict*. A physical law is a device that lets you predict the position of a moving body.
- **Realism**— The law is true always and everywhere, and you are attempting to discover it.

He maps this directly onto machine learning, and the mapping is the frame for the whole conversation.

> The goal of machine learning is to find the rule for classification. It is an instrument for prediction. But for understanding, I need conditional probability.

Everybody today works as an instrumentalist, because the goal is set as finding a decision rule. The realist's version of the same task would be to learn the **conditional probability** (what the probability of each outcome is in a given situation). For prediction you do not need this. For understanding you do.

## Is math poetry?

Lex raises Eugene Wigner's 1960 paper [*The Unreasonable Effectiveness of Mathematics in the Natural Sciences*](https://www.maths.ed.ac.uk/~v1ranick/papers/wigner.pdf). Vapnik's reading of it is that **mathematical structures know something about reality**. Natural scientists look at an equation in order to understand reality, and the same applies to machine learning. If you look carefully at the equations that define conditional probability, you learn more about reality.

What is simple can be very hard to discover. But once discovered, it is beautiful, and it becomes surprising that nobody saw it before.

## Human intuition and ingenuity

Pressed on whether human ingenuity, the moments of brilliance, might leap ahead of the math with the math catching up later, he does not concede it. The best human intuition, in his account, goes into **choosing the axioms**; everything after that is technical work to reach where you have to arrive. And the axioms themselves are not a flash of individual brilliance. They are polished across generations of scientists, and are integral wisdom.

## Interpretation

Asked whether we have, or will have, the tools to describe mathematically what happens in the brain during learning, Vapnik objects to the premise. It would not be a description; it would be an **interpretation**, and your interpretation can be wrong.

His illustration is Antonie van Leeuwenhoek, who built the microscope, kept the instrument secret, and sent reports to the Royal Society. Looking at blood, he saw the cells (he genuinely saw something real) but he described what he saw as a battle between kings and queens, an army fighting. The observation was right and the interpretation was wrong, and the Academy took it seriously because they believed he was right.

Vapnik believes the same thing can happen with the brain.

## The teacher

What he does believe in is human language, and the wisdom compressed into proverbs. The one he returns to: *better than a thousand days of diligent study is one day with a great teacher*. Ask what the teacher actually does and nobody knows— and that, he says, **is** intelligence.

But we know from machine learning that a teacher can do a great deal. What a teacher does, in his formulation, is **introduce invariants**. How the teacher does this he does not know, because the teacher knows reality and can draw a predicate out of it. What is measurable is the effect: using an invariant can cut the number of observations you need by a factor of a hundred.

The piano teacher who says "play like a butterfly" is the example. The phrase is not empty, because it affects your playing. The question Vapnik cares about is what information is actually being transmitted, and in what representation. His answer is that it is a kind of **predicate**— and that this, is the part we do not understand.

## Two mechanisms of convergence

Vapnik's claim is that there are exactly two mechanisms of learning, corresponding to two modes of convergence:

1. **Strong convergence**— the mode classical statistical learning theory uses, requiring the uniform law of large numbers.
2. **Weak convergence**— the mode that lets you use predicates.

The weak mode is where "fly like a butterfly" lives, and it is what makes a predicate immediately affect your solution.

## The duck proverb

The English proverb— *if it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck*— is, in Vapnik's reading, **exactly about predicates**, and the most compact statement of his theory of learning.

Unpack "looks like a duck." You have seen many ducks; that is your training data. From it you have an empirical description. You also have a model that produces a theoretical description. What you want is for the two to coincide.

Now "swims like a duck." Note that you must already know that ducks swim. And notice what the proverb does *not* say:

- You could say *plays chess like a duck*. It is a completely legal predicate. It is also useless, because ducks do not play chess. It carries **zero information**.

So the real question is: **how does a teacher recognize a predicate that is not useless?** Knowing that "swims" is relevant and "plays chess" is not requires knowing ducks, knowing other birds, knowing animals. That knowledge is what selects the predicate.

And this is why current systems need so much data. The proverb gets by with **three predicates**. Existing machine learning uses none, and so needs zillions of examples instead.

## The admissible set of functions

Here Vapnik gives the mechanical picture of what the predicates are doing.

Machine learning starts with a vast collection of candidate functions. You take the training data, form the expectation of what a duck should look like, and **remove every function that does not agree with it**. That shrinks the set. Then you supply a second predicate, which shrinks it again. After that you pick the best remaining function(s). This is standard machine learning— and the reason you do not need many examples is that **every predicate is invented in order to decrease the admissible set of functions**.

An **admissible set of functions** is one with small *capacity*, small *diversity* (small dimension) which nonetheless contains good functions inside it.

This is where the **VC dimension** comes in (Vapnik is the V; Alexey Chervonenkis is the C). A machine picks one function from the admissible set. When the VC dimension is small, you need only a small amount of training data.

So the goal is a set of functions that is **simultaneously small in VC dimension and rich enough to contain good functions**. Get that, and you can identify the right function from few observations.

!!! note "Related pages"
    [Support Vector Machines](../primer_on_ai/introduction_to_machine_learning.md#support-vector-machines) are Vapnik's best-known construction, and [kernel methods](nov_2025_mindscape_336_anil_ananthaswamy_on_the_mathematics_of_neural_nets_and_ai.md#kernel-methods) are how they handle data that is not linearly separable.

## What learning actually is

This produces the reformulation that Vapnik regards as his main point:

> Statistical learning theory does not involve creating admissible set of functions. In classical learning theory everywhere, in 100% of textbooks, the admissible set of functions is given.

That assumption, he argues, tells us nothing, because **creating the admissible set is the hard problem**. Given a continuous, essentially unlimited set of functions, produce a subset with small finite VC dimension that still contains good functions. Classical theory placed this out of consideration; it starts after the hard part is done.

## On Deep Learning

Vapnik's objection is not to the results but to the mode of reasoning, and he opens with an analogy from Churchill's history of the Second World War. In earlier times, when a war ended, kings gathered and negotiated a peace. After the First World War the general public came to power, were greedy, and stripped Germany; it was clear to everyone that this was not peace and would last twenty years, because **they were not professionals**.

He sees the same split in machine learning: mathematicians looking at the problem from a deep mathematical point of view, and computer scientists who mostly do not know the mathematics and work in interpretations instead.

His constructive version is to treat Learning as a mathematical problem, recognize that there is not only the strong mode of convergence but also the weak mode requiring predicates, and work it through. Do that, he claims, and **you find you do not need Deep Learning**.

Lex presses on whether interpretation and play— the neural network as something you throw on the table, the cells imagined as kings and queens- has value as inspiration for where the math eventually leads. Vapnik holds his position: the discussion about deep learning is a discussion about interpretations, not about what you can actually say about things.

## The MNIST challenge

The concrete challenge he closes his talks with, and the one he considers the exact problem of intelligence is the Handwritten Digit Recognition (MNIST). 

Deep learning reports around 99.5% correct using **60,000 training examples**. Can you do the same job with **a hundred times fewer** examples by incorporating invariants, given that you know what the digits 1, 2, 3 *are*?

His own partial attempt: for the digit 3 he would introduce the concept of **horizontal symmetry**, since a 3 has more of it than a 2. Once he has the notion of symmetry, he can invent many mathematical measures of it— horizontal, vertical, diagonal. 

The field's founders noticed immediately that machines needed far more training data than humans, and the question of how to decrease it never went away. Deep learning's answer was more data. But with a good invariant, more data may not even be the relevant axis.

## Can machines think?

On Turing's question, Vapnik notes that Turing described **imitation** and understood perfectly well that he was not describing a thinking computer. He deliberately set up a problem of imitation. What Vapnik objects to is that the field has kept the imitation goal while calling it something else: codes that imitate human activity are applications, not science.

## The open problem

Vapnik separates the field into two stories:

1. **The mathematical story**— given predicates, what you can do. This part, he says, is ready.
2. **How to get the predicates**— the intelligence problem.

And he reduces it to a single formulated question: **why is one teacher better than another?**