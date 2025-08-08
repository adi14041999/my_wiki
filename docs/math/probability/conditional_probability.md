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