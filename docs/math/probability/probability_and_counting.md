# Probability and counting
Mathematics is the logic of certainty; probability is the logic of uncertainty.

## Sample spaces
The mathematical framework for probability is built around sets. Imagine that an experiment is performed, resulting in one out of a set of possible outcomes. Before the experiment is performed, it is unknown which outcome will be the result; after, the result "crystallizes" into the actual outcome.
**The sample space S of an experiment is the set of all possible outcomes of the experiment. An event A is a subset of the sample space S, and we say that A occurred if the actual outcome is in A.**
![image](pebble_world.png)
When the sample space is finite, we can visualize it as Pebble World (figure above). Each pebble represents an outcome, and an event is a set of pebbles. Performing the experiment amounts to randomly selecting one pebble. If all the pebbles are of the same mass, all the pebbles are equally likely to be chosen.

Set theory is very useful in probability, since it provides a rich language for express_ing and working with events. Set operations, especially unions, intersections, and complements, make it easy to build new events in terms of already defined events.

**Example**: 
A coin is flipped 10 times. Writing Heads as **H** and Tails as **T**, a possible outcome is: HHHTHHTTHT.
The sample space is the set of all possible strings of length 10 consisting of **H**'s and **T**'s.
We can (and will) encode **H** as `1` and **T** as `0`, so that an outcome is a sequence:
$$
(s_1, s_2, \dots, s_{10}) \quad \text{with} \quad s_j \in \{0, 1\}
$$
The sample space is the set of all such sequences.

Some Events:

1. Event A_1: the first flip is Heads. As a set:
$$
A_1 =  (1, s_2, \dots, s_{10}) \; \mid \; s_j \in \{0,1\}, \; 2 \leq j \leq 10 
$$
This is a subset of the sample space, so it is indeed an event. Saying that A_1 occurs is equivalent to saying that the first flip is Heads. Similarly, let A_j be the event that the j-th flip is Heads, for:
$$
j = 2, 3, \dots, 10
$$
2. Event B: at least one flip was Heads. As a set:
$$
B = \bigcup_{j=1}^{10} A_j
$$
3. Event C: all the flips were Heads. As a set:
$$
C = \bigcap_{j=1}^{10} A_j
$$
4. Event D: there were at least two consecutive Heads. As a set:
$$
D = \bigcup_{j=1}^{9} \left( A_j \cap A_{j+1} \right)
$$
![image](english_to_sets.png)