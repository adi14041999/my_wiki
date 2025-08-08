# Conditional Probability
## Independent Events
Two events $A$ and $B$ are **independent** if the occurrence of one event does not affect the probability of the occurrence of the other event.

**Mathematical Definition**: Events $A$ and $B$ are independent if and only if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Independence means that knowing whether event $B$ occurred gives us no information about whether event $A$ occurred, and vice versa.

**Example 1: coin flips** Event $A$: First coin flip is heads. Event $B$: Second coin flip is heads. These events are independent because the result of the first flip doesn't affect the second flip. $P(A) = \frac{1}{2}$, $P(B) = \frac{1}{2}$, $P(A \cap B) = \frac{1}{4} = \frac{1}{2} \cdot \frac{1}{2}$ ✓

**Example 2: drawing cards (with replacement)** Event $A$: First card drawn is red. Event $B$: Second card drawn is red (with replacement). These events are independent because we replace the first card. $P(A) = \frac{1}{2}$, $P(B) = \frac{1}{2}$, $P(A \cap B) = \frac{1}{4} = \frac{1}{2} \cdot \frac{1}{2}$ ✓

**Example 3: drawing cards (without replacement) - NOT independent** Event $A$: First card drawn is red. Event $B$: Second card drawn is red (without replacement). These events are NOT independent because drawing a red card first affects the probability of drawing red second.

It's crucial to understand the difference between **independent events** and **disjoint (mutually exclusive) events**.

**Independent Events:**

- Can occur together: $P(A \cap B) = P(A) \cdot P(B) > 0$ (if both $P(A) > 0$ and $P(B) > 0$)

- Knowledge of one event doesn't affect the probability of the other

- Example: Rolling a die twice - getting a 6 on the first roll and a 6 on the second roll

**Disjoint Events:**

- Cannot occur together: $P(A \cap B) = 0$

- If one event occurs, the other cannot occur

- Example: Getting heads and tails on a single coin flip

**Key insight:** If two events $A$ and $B$ are disjoint with $P(A) > 0$ and $P(B) > 0$, then they are **NOT independent**. This is because knowing that $A$ occurred tells us that $B$ definitely did not occur.

Three events $A$, $B$, and $C$ are **mutually independent** if and only if all of the following conditions hold:

**Pairwise independence**:
$P(A \cap B) = P(A) \cdot P(B)$. $P(A \cap C) = P(A) \cdot P(C)$. $P(B \cap C) = P(B) \cdot P(C)$

**Triple independence**: $P(A \cap B \cap C) = P(A) \cdot P(B) \cdot P(C)$

**Important:** Pairwise independence alone is **not sufficient** for mutual independence. All four conditions must be satisfied.

How should you update your beliefs/probabilities based on new evidence? This is a pretty broad question.

## Definition of Conditional Probability

**Conditional probability** is the probability of an event $A$ occurring given that another event $B$ has already occurred.

**Mathematical Definition**: The conditional probability of event $A$ given event $B$ is:

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

provided that $P(B) > 0$.

$P(A | B)$ is read as "the probability of $A$ given $B$". It represents the updated probability of $A$ after we know that $B$ has occurred. **We restrict our sample space to only those outcomes where $B$ occurs.**

**Example**: Consider drawing cards from a standard deck. Let $A$ = "card is an Ace". Let $B$ = "card is red"

$P(A | B) = \frac{P(\text{card is red Ace})}{P(\text{card is red})} = \frac{2/52}{26/52} = \frac{2}{26} = \frac{1}{13}$

This makes sense: among the 26 red cards, only 2 are Aces.

**Key Properties**:

- $0 \leq P(A | B) \leq 1$

- $P(B | B) = 1$ (if $P(B) > 0$)

- If $A$ and $B$ are independent, then $P(A | B) = P(A)$

## Joint Probability

**Joint probability** is the probability that two or more events occur simultaneously.

### Two events

For events $A$ and $B$, the joint probability is:

$$P(A \cap B) = P(A \text{ and } B)$$

**Properties**:

- $0 \leq P(A \cap B) \leq 1$

- $P(A \cap B) = P(B \cap A)$ (commutative)

- $P(A \cap B) = P(A | B) \cdot P(B) = P(B | A) \cdot P(A)$ (multiplication rule)

### Three events

For events $A$, $B$, and $C$, the joint probability is:

$$P(A \cap B \cap C) = P(A \text{ and } B \text{ and } C)$$

**Properties**:

- $0 \leq P(A \cap B \cap C) \leq 1$

- $P(A \cap B \cap C) = P(B \cap A \cap C) = P(C \cap A \cap B)$ (commutative)

- $P(A \cap B \cap C) = P(A | B \cap C) \cdot P(B | C) \cdot P(C)$ (chain rule)

This can be shown as follows:

$P(A | B \cap C) = \frac{P(A \cap B \cap C)}{P(B \cap C)}$.

Thus, $P(A \cap B \cap C) = P(A | B \cap C) \cdot P(B \cap C)$.

Applying multiplication rule to $P(B \cap C)$, we get: $P(B \cap C) = P(B | C) \cdot P(C)$

Substituting into the equation for joint probability, $P(A \cap B \cap C) = P(A | B \cap C) \cdot P(B | C) \cdot P(C)$.

This derivation shows how the chain rule naturally extends the multiplication rule to multiple events.

### n events

For events $A_1, A_2, \ldots, A_n$, the joint probability is:

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1 \text{ and } A_2 \text{ and } \cdots \text{ and } A_n)$$

**Properties**:

- $0 \leq P(A_1 \cap A_2 \cap \cdots \cap A_n) \leq 1$

- $P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1 | A_2 \cap \cdots \cap A_n) \cdot P(A_2 | A_3 \cap \cdots \cap A_n) \cdots P(A_{n-1} | A_n) \cdot P(A_n)$ (chain rule)

- If events are mutually independent, then $P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2) \cdots P(A_n)$

## Law of Total Probability

The **law of total probability** is a fundamental rule that allows us to calculate the probability of an event by considering all possible ways it can occur.

We're breaking down the probability of $A$ by considering how it can occur in each of the different scenarios $B_i$. Since $\{B_1, B_2, \ldots, B_n\}$ is a partition of the sample space, we can write:

$$A = A \cap S = A \cap \left(\bigcup_{i=1}^n B_i\right) = \bigcup_{i=1}^n (A \cap B_i)$$

Since the $B_i$'s are mutually exclusive, the events $(A \cap B_i)$ are also mutually exclusive. Therefore:

$$P(A) = P\left(\bigcup_{i=1}^n (A \cap B_i)\right) = \sum_{i=1}^n P(A \cap B_i)$$

By the definition of conditional probability, $P(A \cap B_i) = P(A | B_i) \cdot P(B_i)$ for each $i$.

Substituting into the previous equation:

$$P(A) = \sum_{i=1}^n P(A | B_i) \cdot P(B_i)$$

**Mathematical Definition**: For any event $A$ and a partition $\{B_1, B_2, \ldots, B_n\}$ of the sample space (i.e., events that are mutually exclusive and exhaustive), we have:

$$P(A) = \sum_{i=1}^n P(A | B_i) \cdot P(B_i)$$

where $B_i \cap B_j = \emptyset$ for all $i \neq j$ (mutually exclusive) and $\bigcup_{i=1}^n B_i = S$ (exhaustive - their union is the entire sample space)

**Example:** Consider two urns. Urn 1 contains 3 red balls and 2 blue balls. Urn 2 contains 1 red ball and 4 blue balls. You randomly select an urn (each with probability $\frac{1}{2}$) and then draw a ball from that urn. What is the probability of drawing a red ball?

Let $A$ = "draw a red ball". Let $B_1$ = "select Urn 1". Let $B_2$ = "select Urn 2". Then, $P(B_1) = \frac{1}{2}$, $P(B_2) = \frac{1}{2}$. $P(A | B_1) = \frac{3}{5}$ (3 red out of 5 balls in Urn 1). $P(A | B_2) = \frac{1}{5}$ (1 red out of 5 balls in Urn 2)

By the law of total probability:

$$P(A) = P(A | B_1) \cdot P(B_1) + P(A | B_2) \cdot P(B_2) = \frac{3}{5} \cdot \frac{1}{2} + \frac{1}{5} \cdot \frac{1}{2} = \frac{3}{10} + \frac{1}{10} = \frac{4}{10} = \frac{2}{5}$$

## Prior and Posterior Probabilities

In many probability problems, we distinguish between **prior** and **posterior** probabilities, which represent our beliefs before and after observing evidence.

**Prior Probability**: The probability of an event before we observe any evidence. This represents our initial belief or knowledge about the event.

**Posterior Probability**: The probability of an event after we observe evidence. This is our updated belief after incorporating new information.

If we have Event $H$ (hypothesis) and Event $E$ (evidence), then:

- **Prior**: $P(H)$ is the prior probability of hypothesis $H$

- **Posterior**: $P(H | E)$ is the posterior probability of hypothesis $H$ given evidence $E$

## Conditional Independence

**Conditional independence** is a concept where two events are independent given knowledge of a third event, even if they might not be independent without that knowledge.

**Mathematical Definition**: Events $A$ and $B$ are **conditionally independent** given event $C$ if and only if:

$$P(A \cap B | C) = P(A | C) \cdot P(B | C)$$

provided that $P(C) > 0$.

Equivalently, this can be written as:

$$P(A | B \cap C) = P(A | C)$$

This means that knowing both $B$ and $C$ gives us no more information about $A$ than knowing just $C$ alone.

**Example**: Consider drawing two cards from a standard deck without replacement.

Let $A$ = "first card is red". $B$ = "second card is red". $C$ = "both cards are the same color"

**Check regular independence**:

- $P(A) = \frac{26}{52} = \frac{1}{2}$

- $P(B) = \frac{26}{52} = \frac{1}{2}$ (by symmetry)

- $P(A \cap B) = \frac{26 \times 25}{52 \times 51} = \frac{25}{102}$

Since $P(A \cap B) = \frac{25}{102} \neq \frac{1}{4} = P(A) \cdot P(B)$, events $A$ and $B$ are **not independent**.

But let's modify this: suppose we draw cards **with replacement**. Then:

- $P(A) = \frac{1}{2}$, $P(B) = \frac{1}{2}$, $P(A \cap B) = \frac{1}{4}$

- So $A$ and $B$ are **independent**

**Check conditional independence given $C$**:
Given that both cards are the same color, we know they're either both red or both black.

- $P(A | C) = P(\text{first red} | \text{both same color}) = \frac{1}{2}$

- $P(B | C) = P(\text{second red} | \text{both same color}) = \frac{1}{2}$

- $P(A \cap B | C) = P(\text{both red} | \text{both same color}) = \frac{1}{2}$

Since $P(A \cap B | C) = \frac{1}{2} \neq \frac{1}{4} = P(A | C) \cdot P(B | C)$, events $A$ and $B$ are **not conditionally independent** given $C$.