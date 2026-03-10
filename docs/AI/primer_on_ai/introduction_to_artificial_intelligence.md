# Introduction to Artificial Intelligence

## Neurons and Hebbian learning

John McCarthy named and helped found the field of AI. His thinking was influenced by the Hixon Symposium on Cerebral Mechanisms in Behavior, which he attended. To appreciate the analogy between brains and machines, it helps to have a minimal picture of how neurons work.

A neuron has dendrites (inputs), a cell body, an axon hillock (where signals are summed), an axon (output cable), and axon terminals that meet other neurons at synapses. The neuron is in a resting state until the combined input at the axon hillock exceeds a threshold. Then it fires—an electrical signal (ion flow) propagates down the axon. The signal is all-or-nothing (same magnitude each time), so in a coarse model the axon output is binary (off or on). At a synapse, the presynaptic terminal releases neurotransmitters from synaptic vesicles. They bind to receptors on the postsynaptic neuron (often compared to a key fitting a lock). Depending on the receptor type, the effect is excitatory (pushes the postsynaptic neuron toward firing) or inhibitory (pushes it away from firing). Motor neurons drive muscle contraction by releasing neurotransmitters that open ion channels in muscle cells. Sensory neurons carry information from sense organs. Interneurons connect other neurons.

![img](neuron.png)

**Hebbian learning** is the idea that when a presynaptic neuron repeatedly helps activate a postsynaptic neuron, the connection between them is strengthened (e.g. more receptors, more neurotransmitter release, or both). The next time, the same presynaptic activity is more likely to trigger the postsynaptic neuron. In this way the brain learns associations— e.g. a “lightning” pattern and a “thunder” pattern that often occur together come to strengthen each other’s pathways. Knowledge is stored in the strength and direction of synaptic connections. The flip side is **“use it or lose it”.** Connections weaken when presynaptic and postsynaptic activity no longer co-occur, which is one basis for forgetting.

## Brains, automata, and the birth of AI

Von Neumann argued that brains and computers were both implementing the same kind of **automaton**—a system that moves through discrete states according to fixed rules and inputs. An automaton is an abstract machine whose next state is determined by its current state and the current input. As a toy illustration, networks of simple threshold units (a neuron fires if weighted input ≥ *T*) can implement logic gates (AND, OR, NOT, etc.), so in principle neuron-like elements can perform logical computation.

John McCarthy famously asked whether computers could be intelligent. Marvin Minsky built the **SNARC** (Stochastic Neural Analog Reinforcement Calculator), which implemented a small network of 40 “neurons” using vacuum tubes and included a form of reinforcement (strengthening successful pathways). That was an early step toward both neural and symbolic approaches to intelligence.

In 1955, McCarthy and others drafted **“A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence.”** The proposal framed seven research themes. (1) Automatic computers—any machine can be simulated by a program. The bottleneck is programming, not hardware. (2) Using language—model thought as manipulating words by rules. Generalize by new words and rules. (3) Neuron nets— how to wire hypothetical neurons to form concepts (partial work by Pitts, McCulloch, Minsky, others). (4) Size of a calculation— need a theory of complexity and efficiency (early work by Shannon, McCarthy). (5) Self-improvement— machines that improve themselves. (6) Abstractions— classify types and how machines could form them from data. Very much like feature extraction. (7) Randomness and creativity— creativity as guided randomness (e.g. hunches) in orderly thought.

The 1956 Dartmouth workshop is often taken as the birth of AI as a named field.

![img](dartmouth.png)

Arthur Samuel worked on checkers-playing programs and wrote “Some Studies in Machine Learning Using the Game of Checkers.” He is credited with coining the term machine learning.

## Rational agents and problem formulation

A **rational agent** is an idealized intelligent system situated in an environment. It receives percepts via sensors and chooses actions via actuators according to an agent function. The agent is given a goal. In this framing, intelligence is figuring out how to reach it. This is a computational definition, not the same as how psychologists define intelligence.

A general problem specification has an initial state, a goal state (or set of goal states), and a state transition function $f(\text{current state}, \text{action}) = \text{next state}$. The agent faces choices along the way. Sometimes each transition has an associated cost. The set of all reachable states and transitions forms a state space graph. A search tree is any tree you get by expanding from the initial state. If a problem can be cast as state space search, solving it reduces to having an algorithm that explores this graph.

Newell and Simon held that the human brain actually does something like this. We perform state space search so quickly on familiar problems that we are not aware of it. Cognitive science, the scientific study of how the mind and brain work, was partly motivated by such ideas.

## Symbols, logic, and continuous state spaces

Newell and Simon also saw computers as symbol manipulators. Binary sequences can stand for anything, not only numbers, and you can define operations on them. Taking a set of axioms (e.g. from Principia Mathematica) and logical manipulation as the basic operations, they treated problem solving as state transitions that could simulate human thinking. They used machines to prove theorems from Principia Mathematica.

The state space need not be discrete. In linear regression you seek a line $y = w_1 x + w_2$ that best fits a cluster of points. With mean squared error $E = \text{MSE}(w_1, w_2)$, each pair $(w_1, w_2)$ is a state and the goal is to minimize $E$. There is no explicit list of actions, but the state space is continuous and from any $(w_1, w_2)$ there are infinitely many possible next states. Calculus supplies a way to move toward the minimum, so the problem can still be viewed as state space search.

![img](sgd.png)

## State space search

Tasks that involve finding a sequence of actions from an initial situation to a goal can be framed as **state space search**— a set of states, a transition function (actions → new states), and one or more goal states. The structure is often drawn as a state space graph. Nodes are states, edges are actions, and solving the problem means finding a path from an initial state to a goal.

- There can be multiple goal states.
- The state space graph can be infinite (e.g. unbounded configurations).
- Paths to the goal may have obstacles or dead ends.

The dynamics are captured by a function $f(\text{current state}, \text{action}) = \text{next state}$. 

A classic example is the Water and Jug Problem (featured in Die Hard 3), which can be solved by viewing jug configurations as states and pours as actions.

**Problem:** You have two jugs with capacities $x$ and $y$ liters (positive integers) and an unlimited water supply. You may fill a jug to capacity, empty a jug, or pour from one jug into the other until the source is empty or the target is full. Determine whether you can measure exactly $z$ liters (i.e. reach a state where one jug contains $z$, or the two jugs together hold $z$).

**State space:** A state is a pair $(a, b)$ where $a$ is the amount in the first jug and $b$ in the second ($0 \leq a \leq x$, $0 \leq b \leq y$). Start at $(0, 0)$. A state is a goal if $a = z$, $b = z$, or $a + b = z$.

**Actions from state $(a, b)$:**

1. Fill jug 1 → $(x, b)$
2. Fill jug 2 → $(a, y)$
3. Empty jug 1 → $(0, b)$
4. Empty jug 2 → $(a, 0)$
5. Pour 1 → 2: pour $\min(a, y - b)$ liters → $(a - \min(a, y - b), b + \min(a, y - b))$
6. Pour 2 → 1: pour $\min(b, x - a)$ liters → $(a + \min(b, x - a), b - \min(b, x - a))$

**Algorithm:** BFS from $(0, 0)$, applying all legal actions from each state and skipping states already visited. If any reached state is a goal, return true. If the queue is exhausted, return false.

```
function CAN_MEASURE(x, y, z):
    if z > x + y or z < 0: return false
    if z == 0: return true
    visited = empty set
    queue = queue containing (0, 0)
    add (0, 0) to visited
    while queue is not empty:
        (a, b) = dequeue(queue)
        if a == z or b == z or a + b == z: return true
        for each next state (a', b') in SUCCESSORS(a, b, x, y):
            if (a', b') not in visited:
                add (a', b') to visited
                enqueue(queue, (a', b'))
    return false

function SUCCESSORS(a, b, x, y):
    next_states = []
    next_states.append((x, b))           // fill jug 1
    next_states.append((a, y))           // fill jug 2
    next_states.append((0, b))           // empty jug 1
    next_states.append((a, 0))           // empty jug 2
    pour = min(a, y - b)
    if pour > 0: next_states.append((a - pour, b + pour))   // pour 1 → 2
    pour = min(b, x - a)
    if pour > 0: next_states.append((a + pour, b - pour))   // pour 2 → 1
    return next_states
```

### Aside on Bidirectional Search

**Bidirectional search** runs two breadth-first searches (BFS) simultaneously— one from the start state forward and one from the goal state backward—stopping when the two fronts meet.

There are several ways to alternate between the two BFS fronts:

1. **Alternate by layer:** Add one full layer to the left BFS tree, then one full layer to the right BFS tree (so both fronts stay the same distance from their starting points).
2. **Strict alternation:** Process one node from the left queue, then one from the right queue, and repeat.
3. **Shorter queue first:** Always take the next node from the tree whose queue is shorter.

The shorter-queue strategy is a heuristic that tends to minimize the total number of nodes processed. You expand the smaller frontier first, which often keeps the combined search more balanced and reduces redundant work compared to fixed alternation.

## Early AI: thinking as search and Moravec’s paradox

Early AI practitioners often equated **thinking with search**. The idea was that intelligence meant exploring a state space efficiently. **Faster search** implied **superior intelligence**. Tasks that machines could solve well in this way— e.g. playing chess, solving puzzles— were the ones that were then regarded as “intelligent.”

At the same time, tasks that humans find effortless—recognizing faces, speaking grammatical sentences, navigating a room—were very hard for computers. **Moravec’s paradox** is the observation that what is easy for humans (perception, movement, language) was difficult for machines, while what is hard for humans (abstract logic, formal math) was relatively easier to automate.

One explanation is evolution and development. Skills that humans have had for tens of thousands of years (vision, motor control, social communication) are deeply ingrained and mostly unconscious. Abstract logical reasoning and formal mathematics are evolutionarily recent and require deliberate effort. Our brains are plastic. With practice, even hard tasks become more automatic, but the “easy” human skills are already highly optimized by evolution.

## Two modes of thinking: System 1 and System 2

The book *Thinking, Fast and Slow* (Kahneman) distinguishes two modes of thinking.

- **System 1:** Fast, intuitive—snap judgments, emotional reactions, pattern recognition. Mostly automatic and low effort.
- **System 2:** Slow, deliberate—logical deduction, calculation, step-by-step analysis. Requires attention and effort.

System 2 is slow in humans in part because working memory is limited and because reasoning is largely serial. We think step by step and do not truly multitask across demanding reasoning tasks.

- State space search is a computational analogue of System 2 thinking—explicit states, actions, and paths. Algorithms (e.g. BFS, A*) explore the space in a structured way.
- Neural networks are well suited to System 1-style tasks—pattern recognition, intuitive prediction, perception. They typically need a lot of data and compute and could not be scaled on early computers.

Neural networks often err (e.g. hallucination, inconsistent logic) doing System 2, so search and classical algorithms remain important. Modern systems often combine both. Neural networks handle System 1 (language, vision, heuristics), while System 2 is delegated to tools such as calculators, code execution, and search (e.g. retrieval, planning).

## Dijkstra's perspective

In many state space search problems, actions are not equal. Each action can have a cost, so the goal is not only to reach a target state but to reach it with minimum total cost. This is the setting for shortest-path methods such as Dijkstra's algorithm. A simple example is route planning on a weighted graph where edges represent travel times and Dijkstra returns the least-cost path from a start node to every reachable node.

In Dijkstra's words, "Computer science is no more about computers than astronomy is about telescopes." The computer is a tool that executes algorithms. The core intellectual object is the algorithm, which predates modern computers by centuries.

## Review of Uninformed Search Strategies

An uninformed search algorithm is given no clue about how close a state is to the goal(s). 

When all actions have the same cost, an appropriate strategy is breadth-first search, in which the root node is expanded first, then all the successors of the root node are expanded next, then their successors, and so on. This is a systematic search strategy that is therefore complete provided the state space either has a solution or is finite. 

When actions have different costs, an obvious choice is Dijkstra’s algorithm by the theoretical computer science community, and uniform-cost search by the AI community. The idea is that while breadth-first search spreads out in waves of uniform depth— first depth 1, then depth 2, and so on— uniform-cost search spreads out in waves of uniform path-cost. Uniform-cost search is cost-optimal, because the first solution it finds will have a cost that is at least as low as the cost of any other node added to the queue of nodes to explore. Uniform cost search considers all paths systematically in order of increasing cost, never getting caught going down a single infinite path (assuming that all action costs are > ǫ > 0). It is therefore complete provided the state space either has a solution or is finite.

Depth-first search always expands the deepest node first. Depth-first search is not cost-optimal; it returns the first solution it finds, even if it is not cheapest. For finite state spaces that are trees it is efficient and complete; for acyclic state spaces it may end up expanding the same state many times via different paths, but will (eventually) systematically explore the entire space. In cyclic state spaces it can get stuck in an infinite loop; therefore some implementations of depth-first search check each new node for cycles. Finally, in infinite state spaces, depthfirst search is not systematic: it can get stuck going down an infinite path, even if there are no cycles. Thus, depth-first search is incomplete. 

## Informed (Heuristic) Search Strategies

An informed search strategy uses domain-specific hints about the location of goals and can find solutions more efficiently than an uninformed strategy. The hints come in the form of a heuristic function, denoted $h(n)$.

$h(n)$ = estimated cost of the cheapest path from the state at node $n$ to a goal state

But how do we decide which node to expand next? A very general approach is called **best-first search**, in which we choose a node, $n$, with minimum value of some evaluation function, $f(n)$. On each iteration we choose a node with minimum $f(n)$ value. We will maintain a min priority queue which will always contain the node with the least $f(n)$.

Concretely, when we expand a node $N$, we look at every node connected to $N$ (e.g. $n_1, n_2, n_3$), compute $f(n_1)$, $f(n_2)$, $f(n_3)$, and insert those nodes into the min priority queue (each with its $f$-value). The queue then holds *all* nodes we have reached but not yet expanded. We always expand next the node at the front of the queue— the one with the smallest $f(n)$. That node need not be a direct neighbor of the node we just expanded; it might be a node that was added in an earlier step. 

**Example:** Consider a small tree: root $A$ with children $B$ and $C$; $C$ has children $D$ and $E$. We expand $A$, compute $f(B)$ and $f(C)$, and add $B$ and $C$ to the queue. Suppose $f(B) = 3$ and $f(C) = 1$. The queue is $[C, B]$. We pop $C$ and expand it. When we expand $C$, we add its children $D$ and $E$ to the queue with, say, $f(D) = 5$ and $f(E) = 4$. The queue is now $[B, E, D]$ (since $f(B)=3$ is less than $f(E)=4$ and $f(D)=5$). So $B$ is still at the front even though we just expanded $C$ and pushed $D$ and $E$. We expand $B$ next— a sibling of $C$, not a child of the node we just expanded. The next node we expand is always the one with minimum $f$ in the queue, which may be an "older" node like $B$ rather than a newly added neighbor.

Below is a figure of a simplified road map of part of Romania, with road distances in miles, which we'll use for our analysis of Informed Search Strategies.

![img](romania.png)

Let's also assume that our destination is Bucharest. Below is the list of straight line distances to Bucharest.

![img](hsld.png)

### Greedy best-first search

Greedy best-first search expands first the node with the lowest $h(n)$ value— the node that appears to be closest to the goal— on the grounds that this is likely to lead to a solution quickly.

So the evaluation function $f(n) = h(n)$.

Let us see how this works for route-finding problems in Romania; we use the straightline distance heuristic, which we will call $h_{SLD}$.

Below is a figure of stages in our greedy best-first tree-like search for Arad to Bucharest with the straight-line
distance heuristic $h_{SLD}$. Nodes are labeled with their h-values.

![img](befs.png)

The solution it found does not have optimal cost. The path via Sibiu and Fagaras to Bucharest is 32 miles longer than the path through Rimnicu Vilcea and Pitesti. This is why the algorithm is called “greedy”— on each iteration it tries to get as close to a goal as it can, but greediness can lead to worse results than being careful. Greedy best-first graph search is complete in finite state spaces, but not in infinite ones.

### $A^*$ search

The flaw with Greedy best-first search is that it does not consider the cost to get to the next node.

The most common informed search algorithm is $A^*$ search (pronounced “A-star search”), a best-first search that uses the evaluation function

$$f(n) = g(n) + h(n)$$

where $g(n)$ is the path cost from the initial state to node $n$, and $h(n)$ is the estimated cost of the shortest path from $n$ to a goal state.

**When $h(v) = 0$ for all $v$: $A^*$ becomes Dijkstra’s algorithm.** If we completely ignore the heuristic by setting $h(v) = 0$ for every vertex $v$, then $f(n) = g(n) + 0 = g(n)$. So we always expand the node with smallest $g$-value (smallest cost from the start so far). That is exactly **Dijkstra’s algorithm** (or **uniform-cost search**). In that case, $A^*$ uses no information about the goal—it is **uninformed** (blind) search. We explore in order of increasing path cost from the source, with no bias toward the goal. So $A^*$ generalizes Dijkstra. With $h = 0$ we get Dijkstra. With a good consistent heuristic we get goal-directed, often much faster, search while still guaranteeing optimality.

Below is a figure showing stages in an $A^*$ search for Bucharest. Nodes are labeled with $f=g+h$. The $h$ values are the straight-line distances to Bucharest.

![img](astar.png)

$A^*$ search is complete, assuming all action costs are > ǫ > 0, and the state space either has a solution or is 
finite (just like BFS and Dijkstra's). Whether $A^*$ is cost-optimal depends on certain properties of the 
heuristic.

#### How do we know $A^*$ search is cost-optimal?

We prove that whenever $A^*$ **expands** a vertex $v$ (i.e. pulls it out of the min priority queue and expands it), the cost $g(v)$ of the path we have from the source $S$ to $v$ is the **optimal** cost from $S$ to $v$. The goal $G$ is just the special case where $v$ is a goal; so when we expand $G$, $g(G)$ is the optimal cost to the goal. We use mathematical induction.

**Base case:** The first vertex expanded is the source $S$. We have $g(S) = 0$, and the optimal cost from $S$ to $S$ is $0$, so the claim holds.

**Inductive step:** Assume that for every vertex we have expanded so far, at the time we expanded it we had the optimal cost to that vertex.

Now, *to expand* a vertex means:

(1) **pull it out** of the min priority queue (it had the smallest $f$-value), and 

(2) **expand from it**— i.e. look at all its successors, compute their $g$ and $f$ values, and add them to the queue. 

So “$v$ is the next vertex we pull out after we pull out $u$ and expand from $u$” means: we expanded $u$, added its successors to the queue, and then the node with minimum $f$ in the queue is $v$ (which might be one of those successors or some other node that was already in the queue). 

We must show that when we expand $v$, $g(v)$ is the optimal cost from $S$ to $v$. Denote by $g^*(v)$ the optimal cost from the source $S$ to any vertex $v$. Then we need to show that for any vertex $v$ that we expand, $g(v) = g^*(v)$.

After some number of steps we have a closed set: vertices that have already been expanded (i.e. captured— pulled out of the priority queue and expanded).

Assume that for every vertex we have expanded so far, at the time we expanded it we had the optimal cost to that vertex. In other words, for any vertex $v'$ that has been expanded so far, $g(v') = g^*(v')$.

Suppose that $v$ is the next vertex to be pulled out from the priority queue. Then $v$ was added to the queue at some earlier step. $v$ is chosen because it has the minimum $f$-value among all vertices currently in the priority queue. $v$ was added to the queue at some earlier step when we expanded some vertex $u$ of which $v$ is an outgoing neighbor. In other words, $u$ is the **parent** of $v$ in the $A^*$ search tree: we had $u$ in the queue, we expanded $u$ (captured $u$), and when we did so we added $v$ to the queue with $g(v) = g(u) + w(u,v)$.

This does *not* mean that $v$ is captured immediately after $u$. We always expand the node with *minimum $f$-value* in the queue. So after we expand $u$, the next node we pull out is whichever node in the queue has the smallest $f$—and that might be $v$, or it might be some other node (e.g. another successor of $u$, or a node that was added to the queue in an even earlier step). So $u$ is the parent of $v$ in the sense that “we added $v$ when we expanded $u$,” but there may be zero or more other vertices captured *between* $u$ and $v$.

$g(v) = g(u) + w(u,v)$. Equivalently, $g(v) = g^*(u) + w(u,v)$, because $u$ was already expanded when we added $v$, and by the inductive hypothesis $g(u) = g^*(u)$.

$v$ is chosen because it has the minimum $f$-value among all vertices currently in the priority queue. 

We need to thus prove in the inductive step that $g(v) = g^*(v)$.

Let's try to do this by **proof by contradiction**.

**Case 1:** 

Suppose there is a different optimal path from $S$ to $v$ that goes through some vertex $y$, and suppose $y$ is still in the priority queue (not yet expanded). Suppose that the parent of $y$ in the search tree is $x$. Suppose also that $x$ is captured. Thus, $g(x) = g^*(x)$ (by the inductive hypothesis, since $x$ has been expanded). There may be none or more vertices along the path from $y$ to $v$. So the optimal path from $S$ to $v$ has the form $S \to \cdots \to x \to y \to \cdots \to v$. Since this is an optimal path, its length is $g^*(v)$.

When $x$ was captured, $y$ was added to the priority queue with $g(y) = g(x) + w(x,y)$, so $f(y) = g(y) + h(y) = g(x) + w(x,y) + h(y)$.

We are about to expand $v$ next, so $v$ has minimum $f$ in the queue. Therefore $f(y) \geq f(v)$. Thus, $g(y) + h(y) \geq g(v) + h(v)$. Since $g(y) = g(x) + w(x,y)$, we have $g(x) + w(x,y) + h(y) \geq g(v) + h(v)$.

Let $c$ be the cost of the path from $y$ to $v$ along the optimal path (through zero or more vertices). Adding $c$ to both sides,

$$g(x) + w(x,y) + h(y) + c \geq g(v) + h(v) + c$$

or

$$g(x) + w(x,y) + c + h(y) \geq g(v) + h(v) + c$$

But $g(x) + w(x,y) + c = g^*(v)$, since we assumed this is the optimal path from $S$ to $v$. Thus,

$$g^*(v) + h(y) \geq g(v) + h(v) + c$$

In the $A^*$ algorithm, we require the heuristic to satisfy $h(y) \leq h(v) + c$. 

If this property holds, then $g^*(v) + h(y) \leq g^*(v) + h(v) + c$. 

Together with $g^*(v) + h(y) \geq g(v) + h(v) + c$, we obtain $g(v) + h(v) + c \leq g^*(v) + h(v) + c$, so $g^*(v) \geq g(v)$. 

But $g^*(v)$ is the optimal cost from $S$ to $v$, so any path has cost at least $g^*(v)$; in particular $g(v) \geq g^*(v)$. 

Thus we have both $g^*(v) \geq g(v)$ and $g(v) \geq g^*(v)$, so $g(v) = g^*(v)$. This contradicts our assumption that there was an optimal path from $S$ to $v$ through a vertex $y$ still in the queue. So no such $y$ exists: when we expand $v$, we have $g(v) = g^*(v)$. 

That inequality is exactly **consistency**: the heuristic at $y$ is at most the cost of the step from $y$ to $v$ plus the heuristic at $v$. In Euclidean geometry this is the triangle inequality (the straight-line distance from $y$ to the goal cannot exceed the distance from $y$ to $v$ plus the distance from $v$ to the goal). So for any consistent heuristic, $h(y) \leq h(v) + c$ holds.

![img](triangle.png)

!!! important "Consistency is required for optimality"
    The heuristic $h$ must be **consistent** to formally guarantee that $A^*$ returns an optimal path. 

**Case 2:** The optimal path directly jumps to $v$ from another captured vertex $x$.

In this case, there must be a second copy of $v$ in the priority queue with a bigger $f$-value (we added all successors of $x$ when we expanded $x$). We are expanding the copy of $v$ that came from $u$ (with minimum $f$), so $f(v)$ through $x$ $\geq$ $f(v)$ through $u$:

$$g(x) + w(x,v) + h(v) \geq g(u) + w(u,v) + h(v)$$

or

$$g(x) + w(x,v) \geq g(u) + w(u,v)$$

Since we are assuming the path through $x$ is optimal, $g(x) + w(x,v) = g^*(v)$. So $g^*(v) \geq g(u) + w(u,v) = g(v)$ (the cost of the path through $u$). 

Thus $g^*(v) \geq g(v)$. But $g^*(v)$ is the optimal cost to $v$, so any path has cost at least $g^*(v)$; in particular $g(v) \geq g^*(v)$. 

So we have both $g^*(v) \geq g(v)$ and $g(v) \geq g^*(v)$, hence $g(v) = g^*(v)$. 

That contradicts the assumption that the optimal path to $v$ goes through $x$ while we expanded $v$ via $u$— it shows the path through $u$ is already optimal, so when we expand $v$ we have $g(v) = g^*(v)$.

In both cases, we got a contradiction, which means the assumption that there exists a different optimal path to $v$ without going through $u$ is categorically **false**.

By induction, whenever $A^*$ expands a vertex $v$, $g(v)$ is optimal. In particular, when we expand a goal $G$, we have found an optimal path to $G$.

Repeating this!!

!!! important "Consistency is required for optimality"
    The heuristic $h$ must be **consistent** to formally guarantee that $A^*$ returns an optimal path. 

#### Consistency implies admissibility

Consider the chain of vertices just before the goal on some path, e.g. $s \to \cdots \to b_3 \to b_2 \to b_1 \to G$, with edge costs $w_1$ (from $b_1$ to $G$), $w_2$ (from $b_2$ to $b_1$), etc. 

We have $h(G) = 0$. 

By consistency, $h(b_1) \leq w_1 + h(G) = w_1$, $h(b_2) \leq w_2 + h(b_1) \leq w_2 + w_1$ (the true cost from $b_2$ to $G$), and so on. 

So for every vertex $v$ on any path to $G$, $h(v)$ is at most the actual cost from $v$ to $G$. 

That is **admissibility**: $h$ never overestimates the cost to the goal. So **consistency $\Rightarrow$ admissibility**. 

Why does admissibility matter? It is because creating an admissible heuristic is much easier than creating a consistent heuristic. For example, for the 8-sliding puzzle, an admissible heuristic can be the number of tiles that are out of place.

The converse is false: a heuristic can be admissible but not consistent.

**Example: admissible but not consistent:** Consider the graph:

- Vertices: $S$, $A$, $B$, $C$, $G$.

- Edges and weights: $S \to A$ (1), $A \to C$ (1), $C \to G$ (3); $S \to B$ (1), $B \to C$ (2).

- So the path $S \to A \to C \to G$ has cost $1+1+3 = 5$ (optimal); $S \to B \to C \to G$ has cost $1+2+3 = 6$.

Heuristic: $h(S)=2$, $h(A)=4$, $h(C)=1$, $h(B)=1$, $h(G)=0$. This $h$ is admissible (each value $\leq$ true cost to $G$). 

But it is **not consistent**: e.g. $h(A)=4$ while $c(A,C)+h(C)=1+1=2$, so $h(A) > c(A,C)+h(C)$.

**Dry run of $A^*$ (tree search: we do not update nodes already in the queue)** 

Start: queue = $\{S\}$, $g(S)=0$, $f(S)=0+2=2$. 

Expand $S$: add $A$ with $g(A)=1$, $f(A)=5$; add $B$ with $g(B)=1$, $f(B)=2$. Queue = $\{B(2), A(5)\}$. 

Expand $B$: add $C$ with $g(C)=3$, $f(C)=4$. Queue = $\{A(5), C(4)\}$. 

Expand $C$: add $G$ with $g(G)=6$, $f(G)=6$. Queue = $\{A(5), G(6)\}$.

Expand $A$: do nothing (no new nodes that change the outcome). Queue = $\{G(6)\}$.

Expand $G$: we have reached the goal; terminate and return the path $S \to B \to C \to G$ with cost $6$. So $A^*$ (in this tree-search style) returns a suboptimal path of cost $6$ instead of the optimal $5$.

#### Summary 

$A^*$ is cost-optimal if and only if the heuristic is **consistent**. Admissibility alone is not enough, as the example above shows.

But in practice, most admissible heuristics tend to be consistent.

#### Some practical examples where $A^*$ is used

1. Apps like Google Maps find a path that minimizes **total travel time** (or distance) between origin and destination. The underlying idea is to treat the road network as a graph and run a shortest-path algorithm. $A^*$ is a natural fit when we can estimate “remaining time to goal.”

2. A robotic arm is described by its **configuration**: the joint angles (and possibly other degrees of freedom). The set of all configurations is **configuration space** (C-space). A point in C-space is one pose of the arm; obstacles in the real world become forbidden regions in C-space. The motion-planning problem is: find a path in C-space from the start configuration to the goal configuration that avoids collisions and often minimizes a cost (e.g. total joint motion or path length in C-space). We can discretize C-space and treat it as a graph: nodes are configurations, edges connect nearby configurations, and edge cost is the distance or "effort" to move between them. $A^*$ then finds a minimum-cost collision-free path. So $A^*$ is widely used in robotics for motion planning in configuration space.

## Rough notes

So far we have considered **state space search** with a clearly defined goal state (e.g. reach Bucharest, measure $z$ liters). Next we consider problems where:

- The **goal is not a single state** but any state satisfying given **properties** (e.g. “no queen attacks another”).
- The **start state can be chosen** (e.g. empty board, or partial assignment).

Such problems are often formulated as **constraint satisfaction**. We have variables and constraints; a goal is any assignment satisfying all constraints.

**Example: N-Queens:** Place $n$ queens on an $n \times n$ chessboard so that no two attack each other.

- **Start state:** e.g. empty board (no queens).
- **State:** any configuration with between $0$ and $n$ queens on the board. The number of such states is $\sum_{k=0}^{n} \binom{n^2}{k}$ (choose any subset of the $n^2$ squares with at most $n$ queens).
- **Goal:** any state with exactly $n$ queens and no pair attacking.
- **Action:** place one queen on an empty square.

Deriving the exact count of *goal* states (valid $n$-queens configurations) is a classical combinatorics problem; the point here is that the state space is finite and the goal is specified by constraints, not by a single target state.
