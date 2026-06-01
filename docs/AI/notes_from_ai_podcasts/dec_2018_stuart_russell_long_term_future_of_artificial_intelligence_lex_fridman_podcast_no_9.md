# Dec 2018. [Stuart Russell- Long-Term Future of Artificial Intelligence | Lex Fridman Podcast no. 9](https://www.youtube.com/watch?v=KsZI5oXBC0k)
9 Dec 2018.

## Meta-reasoning

At Berkeley, Russell worked on **meta-reasoning**: reasoning about reasoning. For a game-playing program, this is the problem of deciding which parts of the search tree are worth exploring, since the tree is enormous, far bigger than the number of atoms in the universe. Programs and humans alike succeed by examining only a small fraction of it: examining the right fraction yields strong play. The guiding principle is that a machine should manage its own computation, thinking about whatever (including avoiding wasteful computation) will most improve its decision quality.

## Two kinds of learning in AlphaGo

Two kinds of learning are at work. First, AlphaGo learns to **evaluate board positions**, with a probably superhuman ability to judge instantly how promising a position is. Human professionals deliberate for five to ten minutes, whereas this stripped-down AlphaGo decides in under a second, because humans lack that level of intuition about Go and must reason the position through.

Second, AlphaGo can look ahead $40$, $50$, or $60$ moves into the future. Searching every possibility that far ahead would be around $10^{200}$ positions, far more than the number of atoms in the universe, so it must be highly selective about what it examines.

!!! note "Evaluation + search approximate MINIMAX"
    These two ingredients mirror how modern game engines achieve strong play. Go, like chess, is far too large to [solve](../primer_on_ai/introduction_to_artificial_intelligence.md#solved-games-and-chess-engines) exactly (i.e. to compute the minimax value at the root). So AlphaGo, like the chess engine Stockfish, **approximates** $\text{MINIMAX}$ rather than computing it: a **learned evaluation function** supplies values for non-terminal positions, and a **selective look-ahead search** explores only the promising lines.

## Why the real world is harder than chess

Several things about chess do not resemble the real world. The rules are known, and the board is completely visible. The real world is mostly not visible from wherever one is sitting, and overcoming partial observability needs qualitatively different algorithms.

The real world also requires planning ahead over timescales of billions or trillions of steps, not in detail, but in commitment. Choosing to do a PhD at Berkeley is a five-year commitment that eventually amounts to roughly a trillion motor-control steps, including every finger movement while typing every character of every paper.

Progress in AI occurs by removing, one by one, the assumptions that make problems easy, such as complete observability. 

As algorithms learn to cope with longer timescales, with uncertainty, and with partial observability, each step magnifies the range of things AI systems can do, roughly by a factor of a thousand.

## Self-driving cars

The first self-driving car drove itself on the freeway- changing lanes and overtaking, back in 1987. More than thirty years later, that is roughly where things still stand. 

The clear bottleneck, then and now, is **perception**. Work on autonomous vehicles at Berkeley in the early-to-mid 1990s included large demonstrations. In simulation with perfect perception, safe driving was achievable for a long time even when other cars misbehaved; but real machine vision for detecting cars and tracking pedestrians could not reach high enough reliability, especially in bad weather, at night, or in rain.

The common mistake is to assume that a successful demo means the job is nearly done.

### Why rules are not enough

Google's early architecture was thoroughly classical: machine vision detected the cars, pedestrians, white lines, and road signs, fed the results into a logical database, and a 1970s-style rule-based expert system decided what to do. The problem was that almost every day brought a situation the rules did not cover (say, a little girl riding a bicycle the wrong way around a traffic circle). Adding more rules never converged.

The deeper question is how to handle a genuinely **novel situation**, where no past case applies and the required reasoning has never been done before. In chess this happens constantly. Each new position is handled by weighing the available actions, their outcomes, and how desirable those outcomes are, and then picking the best one. 

The conclusion in the 1990s was that automated vehicles would likewise need a **look-ahead** capability. But look-ahead is harder for driving than for chess, because humans are less predictable. A chess opponent's intention is known (to win), whereas a driver's is not: turning left, a forgotten turn signal, drunk, or fiddling with the radio. 

The car must estimate the intent of other drivers, forecast how their trajectories might evolve, and choose the safest trajectory— all coupled, since the others react to its own trajectory. The classic case is merging onto the freeway, racing a vehicle already there with uncertainty about who goes first.

### Game Theory in interactions

A key realization when deploying systems in the real world is that an AI system cannot be treated only as something that responds to the world; it is an **agent that others respond to as well**. To drive successfully, a car cannot just do obstacle avoidance and pretend to be invisible; others have to take it into account.

The solutions become quite complicated and lead into game-theoretic analyses. Work at Berkeley on this human–machine interaction has produced interesting, unexpected behavior. When the problem is formulated game-theoretically and the system is left to find the solution, the car sometimes, at a stop sign where no one is going first, backs up a little to signal that the other cars should go, a communication strategy it invented entirely by itself.

### Summary

1. Excellent perception is critical
2. Rule-based systems don't work
3. Look-ahead can help but is not good enough because humans are unpredictable
4. Game theory in human–machine interaction has produced interesting results

## The control problem

It does not take a genius to see that building something smarter than us might be a problem. Alan Turing said as much in a 1951 radio lecture, warning that once machines began to think they would quickly outstrip humanity. The main concern, though, is not super-intelligence as such but the **control problem**: machines pursuing objectives not aligned with ours.

The root difficulty is that we cannot reliably specify those objectives. Human values are extremely hard to put on paper— theoretically possible, but in practice hopeless to enumerate in advance, which is why values are instead transmitted culturally, learned as we grow up. So the standard recipe (build an optimizer and insert an objective) is the wrong approach, because a wrong objective can always slip in, by accident or malice. A machine that treats its objective as gospel truth believes every action in its pursuit is correct, and will keep going even as a human shouts that the world is about to be destroyed. The fix is to make the machine **uncertain about the objective**.

This amounts to teaching machines **humility**: they should know that they do not know what they are supposed to be doing. An uncertain machine is deferential— told "don't do that," it learns something about the true objective and complies, because it wants what we actually want. This implies a different kind of AI: once the objective is no longer assumed known, familiar frameworks (Markov decision processes, goal-based planning, standard games research) no longer apply, and the human becomes part of the problem, since every choice the human makes is evidence about the true objective. The result is inherently game-theoretic, with machine and human coupled together rather than a machine running off alone with a fixed goal.

## Misaligned optimizers: corporations and governments

Some argue that AI systems have already taken over the world in the form of **corporations**: corporations use people as components but are effectively algorithmic machines optimizing an objective, quarterly profit, that is not aligned with the overall well-being of the human race, and they bear much responsibility for our inability to tackle climate change.

More generally, there are many real-world systems where the objective was fixed prematurely and the machine was decoupled from those it is supposed to serve. **Government** is supposed to be a machine that serves people, but it tends to be taken over by people who use it to optimize their own objectives regardless of what people want.

## The absence of oversight in scaling

Because of **scalability**, a single mistake in software or an AI system can harm billions before anyone notices. Just as a faulty drug reaches millions before its effects surface (which is why the FDA exists). Yet for algorithms there is no equivalent (nothing similar to an FDA) oversight.

Social media and **click-through optimization** are a clear example. A simple feedback algorithm that just optimizes click-through sounds reasonable, since people should not be fed ads they do not care about, and it might seem like merely matching ads or news articles to people's preferences. 

But that is not how it works. The algorithm makes more money if it can better predict what people will click on. So the way to maximize click-through is to **modify people to make them more predictable** (for instance by feeding them information that pushes their behavior and preferences toward extremes). People end up at whatever the nearest predictable extreme is, because the machine forces them there. A reasonable argument holds that this, among other things, is contributing to the destruction of democracy.

There was no oversight of this process, no one asking whether applying such an algorithm to five billion people is safe or free of negative effects.

## The three failure modes

One major failure mode is **loss of control**. An AI pursuing an incorrect objective has no incentive to listen, since it believes it already knows the goal— it just executes its computed-optimal strategy, perhaps acquiring resources or resisting interference. This is addressable via the [control problem](#the-control-problem) fix: uncertainty in the objective and deference.

The second failure mode is **misuse**. Even with safe, controllable AI available, a bad actor could deliberately build unsafe AI. **Autonomous weapons** are a clear example of how badly this can go wrong.

The third failure mode is **overuse**, becoming overly dependent on AI—the "WALL-E problem," after the film's passive, obese humans whose machines do everything. This creeps in gradually, like the slow-boiling frog: we hand more and more of civilization's management to machines, sliding from masters of technology to its guests. It is also nearly irreversible. Once people lose the incentive to learn the countless roles that keep civilization running, recovery is very hard.