# Jan 2020. [Daniel Kahneman- Thinking Fast and Slow, Deep Learning, and AI | Lex Fridman Podcast no. 65](https://www.youtube.com/watch?v=UwwBG-MbniY)
14 Jan 2020.

## System 1 and System 2

The real distinction is in *how ideas come to mind*. Some arrive automatically and effortlessly (2 + 2); others require running an algorithm in your head, engaging short-term memory and executive function, and crowding out anything else you might be doing at the same time (27 × 14). **System 1** is the family of automatic, effortless processes. **System 2** is the family of slow, effortful ones. The defining line isn't speed so much as whether mental effort (a genuinely limited resource) is involved.

He traces System 1 to something close to what animals already have: a perceptual system that can anticipate what happens next without being able to explain anything. System 2, by contrast, depends on language and the capacity to manipulate ideas, imagine counterfactuals, and reason conditionally— abilities that need both language and a large brain. Crucially, System 1 is *not* purely instinctive. skills like driving or speaking had to be learned first, through a period where they weren't automatic, before they became so. Once learned, System 1 works by matching a new situation against a previously seen pattern (which is fast and usually right).

For how this same distinction shows up on the AI side— state-space search as a computational analogue of System 2, and neural networks as a System-1-style pattern matcher. See [Two modes of thinking: System 1 and System 2](../primer_on_ai/solving_problems_by_searching.md#two-modes-of-thinking-system-1-and-system-2) in the primer on AI.

## Deep learning as a System 1 achievement

Asked whether these two systems are a useful frame for building AI, Kahneman's view is that current deep learning looks much more like a System 1 product than a System 2 one. It does pattern matching and anticipation (genuinely predictive), but lacks reasoning, causality, or any real representation of meaning. Machine translation systems can produce fluent output while, in a real sense, not knowing what they're talking about.

He's struck by the *speed* of recent progress. The jump from solving chess to solving Go, and then from AlphaGo to AlphaZero, happened faster than almost anyone anticipated. But he flags a specific unsolved problem, credited to AI critic Gary Marcus: children learn from two or three examples, not a million, which means something has to be built into a learner in advance to make it "ready" to learn quickly.

## Grounding, perception, and learning through action

The reason pattern-matching alone falls short, in Kahneman's view, is the absence of **grounding**— being in contact with the world through perception or action, so that words and representations are tied to something rather than floating free. He's genuinely unsure whether a machine needs a body for this, but suspects some form of perceptual grounding is close to essential for accumulating real knowledge about the world.

## How hard is autonomous driving, really?

Lex, working on autonomous vehicles, asks Kahneman how tractable it is to model pedestrians well enough to predict whether they'll cross the road. Kahneman pushes on a specific gap: pedestrians are currently treated by these systems as *obstacles not to be hit*, rather than as agents engaged in something closer to a game-theoretic negotiation. He describes the actual "dance" at a crosswalk: a pedestrian looks the driver in the eye before stepping out (a deliberate signal) and then, notably, looks *away* once committed to crossing, signaling "I'm committed, you now have to yield". The open question is whether a machine needs anything like an understanding of human intention to read this.

## Human-machine collaboration

Asked whether current, System-1-like neural networks could usefully borrow humans for the System 2 parts of a task, Kahneman is skeptical of collaboration as a stable equilibrium: in any human-machine system, once the machine is good enough to genuinely help, it tends to make the human superfluous fairly quickly. The harder case— a machine handing off to a human specifically when it recognizes it's stuck — requires the machine to recognize that it's in a problematic situation without necessarily understanding the problem, which he thinks is very difficult to build.

## Why driving looks easy and isn't

People tend to judge the difficulty of a problem by how hard it feels for *them* personally, which is a poor guide to its actual computational difficulty. For example, driving is easy for a human. Much harder for a machine.

## The experiencing self and the remembering self

Kahneman distinguishes two selves: the **experiencing self**, which simply lives through events in real time (and mostly forgets almost all of them), and the **remembering self**, which periodically evaluates the past and constructs a schematic story about it. The paradox: decisions are governed almost entirely by the remembering self's story, even though the experiencing self is the one that actually lived it.
