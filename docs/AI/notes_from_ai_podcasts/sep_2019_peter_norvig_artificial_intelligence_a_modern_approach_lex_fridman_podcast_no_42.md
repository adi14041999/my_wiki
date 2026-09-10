# Sep 2019. [Peter Norvig- Artificial Intelligence- A Modern Approach | Lex Fridman Podcast no. 42](https://www.youtube.com/watch?v=_VPxEcT_Adc&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
30 Sep 2019.

## How the textbook changed across editions

*Artificial Intelligence: A Modern Approach*, written with [Stuart Russell](dec_2018_stuart_russell_long_term_future_of_artificial_intelligence_lex_fridman_podcast_no_9.md), is the field-defining textbook. Norvig traces its changes across four editions.

### Compute changed the technology choices

The shift from the first to the second and third editions was driven largely by the **rise of computing power**, and it played out in an unexpected direction.

In the first edition they presented propositional (predicate) logic, then argued that it only goes so far: you quickly end up with millions of short expressions that could not possibly fit in memory. Then they realized the premise had quietly expired. Propositional logic is actually rather nice, because there are very fast SAT solvers (SAT is the boolean satisfiability problem: given a propositional formula, is there any assignment of true/false to its variables that makes the whole thing true?), and millions (or even billions) of expressions now fit easily into memory.

So a change in hardware changed which representation was the right one. Norvig sees the same thing happening again with GPUs and with more specific machinery like [TPUs](may_2019_chris_lattner_compilers_llvm_swift_tpu_and_ml_accelerators_lex_fridman_podcast_no_21.md#what-a-tpu-is) and custom ASICs for deep learning.

### From optimizing utility to choosing it

The more interesting change is philosophical. All three of the first editions defined AI as **maximizing expected utility**: you tell me your utility function, and here are 27 chapters of techniques for optimizing it.

The fourth edition takes a different view. Maybe the optimization is the easy part, and **the hard part is deciding what the utility function should be**. What do I want, and if I am a collection of agents or a society, what do *we* want?

## Learning what humans want

Asked whether human values can be encoded into a utility function purely from data, Norvig says there is no single answer yet, only the beginnings of one.

The main technique is **inverse reinforcement learning**. Ordinary [reinforcement learning](dec_2018_pieter_abbeel_deep_reinforcement_learning_lex_fridman_podcast_no_10.md) takes actions, receives rewards, and figures out what actions to take. Inverse RL runs the other way: you observe somebody taking actions and infer what they must have been trying to do.

The obvious restriction is that **people frequently act against their own interests**. Plenty of human actions are self-destructive or suboptimal, and you do not want to learn those. You want to recover the actions a person *should* have taken rather than the ones they did. That gap is a standing challenge for the field.

## What was right in expert systems

Asked whether the symbolic systems of the 1980s left anything worth reviving, Norvig says representation and reasoning are crucial. Sometimes you simply **do not have enough data to learn de novo**, so you need some representation (programmed in, told, or otherwise acquired) and the ability to take steps of reasoning over it.

He identifies two specific failures of the old approach.

**Atomic symbols were a poor match for the world.** They work beautifully for something like a triangle, which has necessary and sufficient conditions. But the real world is messy and does not have sharp edges, while atomic symbols do.

**Reasoning was universal, with no guidance about where to apply it.** Being applicable anywhere sounds like a virtue, but it means nothing stops you from applying a rule repeatedly.

## Explainability versus trust

Norvig prefers to frame the problem as **trust, validation and verification**, with explanation as one tool serving those goals rather than the goal itself.

Explanations matter. If I am denied a loan I want to know why, and under GDPR in Europe you are entitled to that. But **explanation alone is not enough**, because an explanation carries no guarantee of relating to reality. The bank can tell me I lacked collateral; that may be true, or the real reason may have been my religion. I cannot tell from the explanation. Note that this is equally true whether the decision was made by a computer or a person.

So he wants more.

- **Conversation rather than a single output:** Being able to go back and forth: you gave this explanation, but what about this, what would have happened if that had been different, what would I need to change?
- **Testing across cases:** From my case alone I cannot tell whether the decision turned on collateral or on skin color. Across all the cases, the pattern is detectable.
- **Adversarial testing:** We thought we were close to human-level performance on ImageNet, and then adversarial images showed that part of it is nothing like human performance.

### Seduced by low-dimensional metaphors

This is the sharpest idea in the conversation. Norvig thinks part of the problem is that **we are seduced by our low-dimensional metaphors**.

The textbook picture is a flat 2D space, mapped out: cat over here, dog over there, maybe a small ambiguous region in the middle, but mostly covered. If you believe that metaphor, you conclude we are nearly there and that adversarial examples are a handful of edge cases.

The right picture is different. It is a **million-dimensional space**, and "cat" is a thin string winding through it on some crazy path. Step a little off that path in any direction and you are in nowhere-land, with no idea what happens.

That is less an explanation than an understanding of what the models actually are. And it is the understanding you need before you can start fixing anything. See [the curse of dimensionality](nov_2025_mindscape_336_anil_ananthaswamy_on_the_mathematics_of_neural_nets_and_ai.md#the-curse-of-dimensionality) for why intuitions from two or three dimensions mislead so badly.

## Human-level intelligence

Norvig resists the framing on two counts.

**It is not one thing.** There are many different tasks and capabilities, and no single threshold.

**It should not be the goal.** He would not want to build a calculator that multiplies at human level- that would be a step backwards. For many things we should aim far beyond human level; for some, human level is the right target; for others we should not bother, since humans already do them.

What he prefers to focus on is **what makes a useful tool** — and in some cases reaching human level is exactly what makes a tool cross the threshold into usefulness. Personal assistants are a good example: the model we are aiming at is having a real conversation.

## What counts as a test of intelligence

Norvig is impressed all the time, and thinks conversation matters. But conversational tests are easy to fool.

His reading of Turing is the useful part. He suspects Turing's point was **not that a conversation is a good test**, but that *having a test is the right approach*. Rather than letting philosophers declare AI impossible, you set a test and let the result decide. That does not commit you to conversation specifically, and devising better tests as the technology evolves is the right way to proceed.

## Threats

Norvig thinks about dangers, on the principle that any powerful technology can be used for bad as well as good. But he is **not worried about the robot apocalypse** or Terminator scenarios.

What concerns him:

- **Employment and inequality:** Change in employment, and whether we can react fast enough to deal with it. People are already disgruntled about income inequality, and automation could accelerate those problems.
- **Weaponization:** Powerful technologies always can be used as weapons (robots, drones). Some of that involves AI and much of it does not.
- **The wider field of threats:** He is not sure whether an autonomous drone or the availability of CRISPR is the worse one. We have many threats to face; some involve AI and some do not.

On whether he is optimistic that technology also alleviates these threats: it is hard to predict. Society has survived nuclear weapons so far, followed immediately by the observation that only the societies which survived are around to have this conversation, so there may be some **survivorship bias** in the reassurance.