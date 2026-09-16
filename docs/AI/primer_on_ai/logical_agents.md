# Logical Agents

## Knowledge-Based Agents

The problem-solving agents encountered in previous chapters ([search](solving_problems_by_searching.md), [constraint satisfaction](constraint_satisfaction_problems.md), and [adversarial games](adversarial_search_and_games.md)) know things, but only in a very limited, inflexible sense. They know what actions are available and what the result of performing a specific action from a specific state will be, but they don't know general facts. A route-finding agent doesn't know that it is impossible for a road to be a negative number of kilometers long. An 8-puzzle agent doesn't know that two tiles cannot occupy the same space.

The central component of a knowledge-based agent is its **knowledge base ($\text{KB}$)**. A knowledge base is a set of **sentences**. "Sentence" here is a technical term, related but not identical to sentences in English or other natural languages. Each sentence is expressed in a **knowledge representation language** and represents some assertion about the world. When a sentence is taken as given without being derived from other sentences, we call it an **axiom**.

There must be a way to add new sentences to the knowledge base and a way to query what is known. The standard names for these operations are $\text{TELL}$ and $\text{ASK}$, respectively.

```
function KB-AGENT(percept) returns an action
    persistent: KB, a knowledge base
                t, a counter, initially 0, indicating time

    TELL(KB, MAKE-PERCEPT-SENTENCE(percept, t))
    action ← ASK(KB, MAKE-ACTION-QUERY(t))
    TELL(KB, MAKE-ACTION-SENTENCE(action, t))
    t ← t + 1
    return action
```

The agent maintains a knowledge base $\text{KB}$, which may initially contain some **background knowledge**.

Each time the agent program is called, it does three things. First, it $\text{TELL}$s the knowledge base what it perceives. Second, it $\text{ASK}$s the knowledge base what action it should perform— in the process of answering this query, extensive reasoning may be done about the current state of the world, the outcomes of possible action sequences, and so on. Third, it $\text{TELL}$s the knowledge base which action was chosen, and returns the action so it can be executed.

The details of the representation language are hidden inside three helper functions. $\text{MAKE-PERCEPT-SENTENCE}$ constructs a sentence asserting that the agent perceived the given percept at time $t$. $\text{MAKE-ACTION-QUERY}$ constructs a sentence asking what action should be performed at time $t$. $\text{MAKE-ACTION-SENTENCE}$ constructs a sentence asserting that the chosen action was executed. The details of the inference mechanisms are hidden inside $\text{TELL}$ and $\text{ASK}$.

For example, an automated taxi might have the goal of taking a passenger from San Francisco to Marin County and might know that the Golden Gate Bridge is the only link between the two locations. We can then expect it to cross the Golden Gate Bridge, because it knows that doing so will achieve its goal. Notice that this analysis is independent of how the taxi works at the implementation level. It doesn't matter whether its geographical knowledge is stored as linked lists or pixel maps, or whether it reasons by manipulating strings of symbols in registers or by propagating noisy signals through a network of neurons.

## Logic

We said that knowledge bases consist of sentences. These sentences are expressed according to the **syntax** of the representation language. **Semantics** defines the meaning of sentences. Specifically, it specifies what each sentence asserts about the world. A sentence is true or false relative to a **possible world** (also called a **model**). The semantics tells you, for any given sentence $\alpha$ and any given model $m$, whether $\alpha$ holds in $m$. This is what distinguishes a knowledge representation language from mere syntax: syntax is the form of sentences, semantics is what they mean. For example, the semantics for arithmetic specifies that the sentence $x + y = 4$ is true in a world where $x = 2$ and $y = 2$, but false in a world where $x = 1$ and $y = 1$.

When we need to be precise, we use the term **model** in place of "possible world." Whereas possible worlds might be thought of as (potentially) real environments that the agent might or might not be in, models are mathematical abstractions, each of which has a fixed truth value (true or false) for every relevant sentence.

Informally, we might think of a possible world as having $x$ men and $y$ women sitting at a table playing bridge, where the sentence $x + y = 4$ is true when there are four people in total. Formally, the possible models are all possible assignments of non-negative integers to the variables $x$ and $y$. Each such assignment determines the truth of any arithmetic sentence whose variables are $x$ and $y$.

If a sentence $\alpha$ is true in model $m$, we say that $m$ **satisfies** $\alpha$. We use the notation $M(\alpha)$ to mean the set of all models of $\alpha$.

Now that we have a notion of truth, we are ready to talk about logical reasoning. This involves the relation of **logical entailment** between sentences— the idea that a sentence follows logically from another. In mathematical notation, we write $\alpha \models \beta$ to mean that the sentence $\alpha$ entails the sentence $\beta$. The formal definition is: $\alpha \models \beta$ if and only if, in every model in which $\alpha$ is true, $\beta$ is also true.

The definition of entailment can be applied to derive conclusions— that is, to carry out **logical inference**. Given a knowledge base $\text{KB}$ and a query sentence $\alpha$, the goal of inference is to determine whether $\text{KB} \models \alpha$: does everything the agent knows logically guarantee that $\alpha$ is true? An inference algorithm that only derives entailed sentences is called **sound**— it never draws a false conclusion. One that can derive every entailed sentence is called **complete**— it never misses a true one. Together, soundness and completeness characterize what it means for an inference procedure to be both trustworthy and thorough. 

If an inference algorithm $i$ can derive $\alpha$ from $\text{KB}$, we write $\text{KB} \vdash_i \alpha$, pronounced "$\alpha$ is derived from $\text{KB}$ by $i$" or "$i$ derives $\alpha$ from $\text{KB}$."

## Propositional Logic: A very simple Logic

### Syntax

The syntax of propositional logic defines the allowable sentences. The atomic sentences consist of a single proposition symbol. Each such symbol stands for a proposition that can be true or false.

There are two proposition symbols with fixed meanings: $\text{True}$ is the always-true proposition and $\text{False}$ is the always-false proposition.

Complex sentences are constructed from simpler sentences using parentheses and **logical connectives**. There are five connectives in common use:

- $\lnot$ **(not).** A sentence such as $\lnot P$ is called the **negation** of $P$.

- $\land$ **(and).** A sentence whose main connective is $\land$, such as $P \land Q$, is called a **conjunction**; its parts are the **conjuncts**.

- $\lor$ **(or).** A sentence whose main connective is $\lor$, such as $(P \land Q) \lor R$, is a **disjunction**; its parts are the **disjuncts**.

- $\Rightarrow$ **(implies).** A sentence such as $(P \land Q) \Rightarrow \lnot R$ is called an **implication** (or **conditional**). Implications are also known as **rules** or if–then statements. The implication symbol is sometimes written as $\supset$ or $\rightarrow$.

- $\Leftrightarrow$ **(if and only if).** A sentence such as $P \Leftrightarrow \lnot Q$ is a **biconditional**.

### Semantics

Having specified the syntax of propositional logic, we now specify its semantics. The semantics defines the rules for determining the truth of a sentence with respect to a particular model. In propositional logic, a model simply sets the truth value ($\text{true}$ or $\text{false}$) for every proposition symbol. 

The Truth tables for the five logical connectives are shown below.

| $P$ | $Q$ | $\lnot P$ | $P \land Q$ | $P \lor Q$ | $P \Rightarrow Q$ | $P \Leftrightarrow Q$ |
|-------|-------|-----------|-------------|------------|-------------------|----------------------|
| false | false | true      | false       | false      | true              | true                 |
| false | true  | true      | false       | true       | true              | false                |
| true  | false | false     | false       | true       | false             | false                |
| true  | true  | false     | true        | true       | true              | true                 |

### A simple knowledge base

Now that we have defined the semantics for propositional logic, we can construct a knowledge base. Consider the following three proposition symbols:

- $R$: "It is raining."
- $U$: "I have an umbrella."
- $W$: "I get wet."

We populate $\text{KB}$ with two sentences (axioms):

1. $R$— it is raining.
2. $R \land \lnot U \Rightarrow W$— if it is raining and I have no umbrella, then I get wet.

**Syntax:** Both sentences are well-formed according to the syntax of propositional logic. They are built from atomic symbols using the connectives $\land$, $\lnot$, and $\Rightarrow$.

**Semantics:** Consider the model $m = \{R = \text{true},\ U = \text{false},\ W = \text{true}\}$. Evaluating each sentence of $\text{KB}$ in $m$:

- $R$ is $\text{true}$ in $m$. ✓
- $R \land \lnot U$ is $\text{true} \land \text{true} = \text{true}$, so $R \land \lnot U \Rightarrow W$ is $\text{true} \Rightarrow \text{true} = \text{true}$. ✓

Both sentences are satisfied, so $m$ is a model of $\text{KB}$.

### A simple inference procedure

Our goal now is to decide whether $\text{KB} \models \alpha$ for some sentence $\alpha$.

!!! note "Axioms must be true"
    The sentences in $\text{KB}$ are asserted to be true— they are not hypotheses to be tested. This is what makes them axioms. The only models that matter for inference are those in which every axiom is satisfied; assignments that violate any axiom are not models of $\text{KB}$ and are ignored. Entailment is therefore always relative to what the agent has committed to believing.

Take $\alpha = W$ ("I get wet") as our query. We want to know whether $W$ is **true in every model of $\text{KB}$**.

$\text{KB}$ has two axioms: $R$ and $R \land \lnot U \Rightarrow W$. $R$ must be $\text{true}$ (otherwise the first axiom fails). Given $R = \text{true}$, the second axiom says: if $U = \text{false}$, then $W$ must be $\text{true}$.

- If $U = \text{false}$: the antecedent $R \land \lnot U$ is $\text{true}$, so $W$ must be $\text{true}$ for the implication to hold. The only satisfying assignment is $W = \text{true}$.
- If $U = \text{true}$: the antecedent $R \land \lnot U$ is $\text{false}$, so the implication holds regardless of $W$. Both $W = \text{true}$ and $W = \text{false}$ are consistent with $\text{KB}$.

So $M(\text{KB})$ includes models where $W = \text{false}$ (when $U = \text{true}$). Therefore $\text{KB} \not\models W$.

If we add a third axiom $\lnot U$ (I do not have an umbrella) to $\text{KB}$, then every remaining model has $R = \text{true}$ and $U = \text{false}$, forcing $W = \text{true}$ in all of them. Now $\text{KB} \models W$.

## Propositional Theorem Proving

Two sentences $\alpha$ and $\beta$ are **logically equivalent** if they are true in the same set of models. We write this as $\alpha \equiv \beta$, which is equivalent to saying $M(\alpha) = M(\beta)$, or equivalently that $\alpha \models \beta$ and $\beta \models \alpha$.

A sentence is **valid** if it is true in all models. For example, the sentence $P \lor \lnot P$ is valid. Valid sentences are also known as **tautologies**— they are necessarily true.

A sentence is **satisfiable** if it is true in, or satisfied by, some model. Satisfiability can be checked by enumerating the possible models until one is found that satisfies the sentence.

Many problems in computer science are really satisfiability problems. For example, all the [constraint satisfaction problems](constraint_satisfaction_problems.md) ask whether the constraints are satisfiable by some assignment.

### Inference rules

Rather than checking entailment by enumerating all models, we can derive new sentences directly using **inference rules**— patterns that produce conclusions from premises. 

#### Modus Ponens

The most fundamental is **modus ponens**:

$$\frac{\alpha \Rightarrow \beta, \quad \alpha}{\beta}$$

Read: if we know $\alpha \Rightarrow \beta$ and we know $\alpha$, we can conclude $\beta$. The sentences above the line are the premises; the sentence below is the conclusion.

Modus ponens is **sound**: whenever both premises are true in a model, the conclusion is also true. It is not on its own **complete**. There are entailed sentences that cannot be reached by modus ponens alone. However, combined with other inference rules it forms the basis of complete proof systems for propositional logic.

**Example:** Does $P \land \lnot P$ entail $Q$?

Recall that $\alpha \models \beta$ if and only if $M(\alpha) \subseteq M(\beta)$. The sentence $P \land \lnot P$ is a **contradiction**. It requires $P$ to be both $\text{true}$ and $\text{false}$ simultaneously, which is impossible. Therefore $M(P \land \lnot P) = \emptyset$: there are no models in which it holds.

The empty set is a subset of every set, so $\emptyset \subseteq M(Q)$ for any sentence $Q$ whatsoever. It follows that $P \land \lnot P \models Q$. A contradiction entails everything.

This is known as the **principle of explosion** (*ex contradictione quodlibet*: "from a contradiction, anything follows"). It is why keeping a knowledge base consistent is critical: if even one contradiction enters $\text{KB}$, then $\text{KB} \models \alpha$ for every sentence $\alpha$, making the KB useless as a reasoning tool.

!!! note "Entailment vs. derivability"
    - $\models$ **(entailment)** is a **semantic** relation. $\text{KB} \models \alpha$ means $\alpha$ is true in every model where $\text{KB}$ is true— it's a statement about truth in the world, independent of any procedure.
    - $\vdash$ **(derivability)** is a **syntactic** relation. $\text{KB} \vdash_i \alpha$ means a specific inference algorithm $i$ can produce $\alpha$ from $\text{KB}$ by mechanically applying inference rules.

**Example:** The vault wiping protocol

An automated security system has triggered a lockdown on a secure vault. You are a digital forensics analyst trying to determine whether the automated wiping protocol was initiated. You extract the following six premises from the system log, using the proposition symbols $F$ (firewall breached), $A$ (alarm sounds), $S$ (silent alert sent), $P$ (main power grid shuts down), $G$ (backup generator activates), $L$ (vault door locked), $W$ (wiping protocol initiated).

| # | Natural language | Formal sentence |
|---|-----------------|-----------------|
| 1 | If $F$ then $A$ or $S$, but not both (XOR) | $F \Rightarrow (A \lor S) \land \lnot(A \land S)$ |
| 2 | The firewall was breached | $F$ |
| 3 | The alarm sounds iff the power grid shuts down | $A \Leftrightarrow P$ |
| 4 | If a silent alert is sent, the backup generator activates | $S \Rightarrow G$ |
| 5 | The main power grid did not shut down | $\lnot P$ |
| 6 | If the generator activates and the vault door is locked, the wiping protocol runs; the door remained locked | $G \land L \Rightarrow W$ and $L$ |

**Step 1.** From premise 3 ($A \Leftrightarrow P$) and premise 5 ($\lnot P$):

$A \Leftrightarrow P$ is an axiom, so it must be $\text{true}$ in every model of $\text{KB}$. A biconditional is $\text{true}$ only when both sides share the same truth value. Since premise 5 fixes $P = \text{false}$, the only assignment that keeps the biconditional $\text{true}$ is $A = \text{false}$.

| $A$ | $P$ | $A \Leftrightarrow P$ | Valid model of $\text{KB}$? |
|-------|-------|----------------------|---------------------------|
| false | false | true                 | ✓ (both axioms satisfied) |
| true  | false | false                | ✗ (violates axiom 3)      |

Only the first row survives. In every model of $\text{KB}$, $A = \text{false}$.

$$\lnot P,\quad A \Leftrightarrow P \;\vdash\; \lnot A$$

**Step 2.** From premise 1 ($F \Rightarrow (A \lor S) \land \lnot(A \land S)$) and premise 2 ($F$), by modus ponens:

$$(A \lor S) \land \lnot(A \land S)$$

We now know $\lnot A$ (Step 1). Substituting into the XOR: $A \lor S$ must be $\text{true}$ and since $A$ is $\text{false}$, $S$ must be $\text{true}$.

$$\lnot A,\quad (A \lor S) \land \lnot(A \land S) \;\vdash\; S$$

**Step 3.** From premise 4 ($S \Rightarrow G$) and $S$ (Step 2), by modus ponens:

$$S \Rightarrow G,\quad S \;\vdash\; G$$

**Step 4.** From premise 6 ($G \land L \Rightarrow W$), $G$ (Step 3), and $L$ (given in premise 6), by $\land$-introduction then modus ponens:

$$G \land L \Rightarrow W,\quad G \land L \;\vdash\; W$$

**Conclusion:** $\text{KB} \models W$. The automated wiping protocol **was initiated**. Each step used modus ponens on a sound, consistent knowledge base, so the conclusion is guaranteed to be true in every model of $\text{KB}$.

**Combined truth table (the tedious way; without using inference rules):** Premises 2, 5, and the $L$ clause of premise 6 fix $F = \text{true}$, $P = \text{false}$, and $L = \text{true}$ in every model of $\text{KB}$. We enumerate all $2^4 = 16$ combinations of the remaining variables $A$, $S$, $G$, $W$ and check which rows satisfy all four remaining premises simultaneously. A check (✓) means the premise holds in that row; a cross (✗) means it fails and the row is not a model of $\text{KB}$.

| $A$ | $S$ | $G$ | $W$ | P1: $(A \oplus S)$ | P3: $A \Leftrightarrow \text{false}$ | P4: $S \Rightarrow G$ | P6: $G \Rightarrow W$ | Model of $\text{KB}$? |
|-----|-----|-----|-----|--------------------|--------------------------------------|-----------------------|-----------------------|----------------------|
| F | F | F | F | ✗ | ✓ | ✓ | ✓ | ✗ |
| F | F | F | T | ✗ | ✓ | ✓ | ✓ | ✗ |
| F | F | T | F | ✗ | ✓ | ✓ | ✗ | ✗ |
| F | F | T | T | ✗ | ✓ | ✓ | ✓ | ✗ |
| F | T | F | F | ✓ | ✓ | ✗ | ✓ | ✗ |
| F | T | F | T | ✓ | ✓ | ✗ | ✓ | ✗ |
| F | T | T | F | ✓ | ✓ | ✓ | ✗ | ✗ |
| **F** | **T** | **T** | **T** | **✓** | **✓** | **✓** | **✓** | **✓** |
| T | F | F | F | ✓ | ✗ | ✓ | ✓ | ✗ |
| T | F | F | T | ✓ | ✗ | ✓ | ✓ | ✗ |
| T | F | T | F | ✓ | ✗ | ✓ | ✗ | ✗ |
| T | F | T | T | ✓ | ✗ | ✓ | ✓ | ✗ |
| T | T | F | F | ✗ | ✗ | ✗ | ✓ | ✗ |
| T | T | F | T | ✗ | ✗ | ✗ | ✓ | ✗ |
| T | T | T | F | ✗ | ✗ | ✓ | ✗ | ✗ |
| T | T | T | T | ✗ | ✗ | ✓ | ✓ | ✗ |

Exactly one row satisfies all premises: $A = \text{false}$, $S = \text{true}$, $G = \text{true}$, $W = \text{true}$. Since $W = \text{true}$ in the unique model of $\text{KB}$, we confirm $\text{KB} \models W$.