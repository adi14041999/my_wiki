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

Informally, **logic** is the study of what follows from what. Given a set of things you know to be true, logic tells you what else must be true— not by guessing or observing, but by the structure of the statements themselves. It has three parts: the words and symbols we use to express facts (**syntax**), what those facts actually mean in the world (**semantics**), and the rules for deriving new facts from existing ones (**inference**).

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

## First-Order Logic

Propositional logic sufficed to illustrate the basic concepts of logic, inference, and knowledge based agents. Unfortunately, propositional logic is limited in what it can say.

### Syntax and Semantics of First-Order Logic

Models of a logical language are the formal structures that constitute the possible worlds under consideration. In First-Order Logic, the domain of a model is the set of objects or domain elements it contains. The domain is required to be nonempty— every possible world must contain at least one object.

**Example: models and domains:** Suppose we are reasoning about a small office. One possible model $m_1$ has the domain $\{ \text{Alice}, \text{Bob}, \text{Room101}, \text{Laptop} \}$. A second model $m_2$ has a larger domain $\{ \text{Alice}, \text{Bob}, \text{Carol}, \text{Room101}, \text{Laptop} \}$, representing a possible world where a third employee exists. A third model $m_3$ might have a completely different domain $\{ \text{Dave}, \text{Server}, \text{Room202} \}$, representing an entirely different office. Each possible world is a different model with its own domain.

The objects in the model may be related in various ways. Formally speaking, a relation is just the set of tuples of objects that are related. A tuple is a collection of objects arranged in a fixed order and is written with angle brackets surrounding the objects.

**Example: tuples:** Suppose our domain contains three people: Alice, Bob, and Carol. The tuple $\langle \text{Alice}, \text{Bob} \rangle$ represents an ordered pair. It is different from $\langle \text{Bob}, \text{Alice} \rangle$ because order matters.

**Example: a relation:** The relation $\text{Knows}$ might capture who knows whom. If Alice knows Bob and Bob knows Carol, the relation is the set of tuples:

$$\text{Knows} = \{ \langle \text{Alice}, \text{Bob} \rangle,\ \langle \text{Bob}, \text{Carol} \rangle \}$$

We say $\text{Knows}(\text{Alice}, \text{Bob})$ is true (the tuple is in the relation) and $\text{Knows}(\text{Alice}, \text{Carol})$ is false (the tuple is not). Relations can involve any number of objects: a unary relation (one object) such as $\text{IsAdult} = \{ \langle \text{Alice} \rangle,\ \langle \text{Carol} \rangle \}$ captures a property of individual objects, while a ternary relation such as $\text{Introduced}(\text{Alice}, \text{Bob}, \text{Carol})$ (Alice introduced Bob to Carol) involves three.

**Every model must provide the information required to determine if any given sentence is true or false.**

**Example:** Suppose we want to describe a small university department. We define a model $m$ as follows.

**Domain:** $\Delta = \{ \text{Alice}, \text{Bob}, \text{Carol}, \text{CS101}, \text{CS201} \}$

Alice, Bob, and Carol are people; CS101 and CS201 are courses.

**Relations:**

- $\text{Professor} = \{ \langle \text{Alice} \rangle,\ \langle \text{Bob} \rangle \}$. Alice and Bob are professors (unary; captures a property of individuals).
- $\text{Student} = \{ \langle \text{Carol} \rangle \}$. Carol is a student.
- $\text{Course} = \{ \langle \text{CS101} \rangle,\ \langle \text{CS201} \rangle \}$. CS101 and CS201 are courses.
- $\text{Teaches} = \{ \langle \text{Alice}, \text{CS101} \rangle,\ \langle \text{Bob}, \text{CS201} \rangle \}$. Alice teaches CS101, Bob teaches CS201 (binary relations).
- $\text{EnrolledIn} = \{ \langle \text{Carol}, \text{CS101} \rangle,\ \langle \text{Carol}, \text{CS201} \rangle \}$. Carol is enrolled in both courses.
- $\text{Prerequisite} = \{ \langle \text{CS101}, \text{CS201} \rangle \}$. CS101 is a prerequisite for CS201.

**Querying the model:** Given $m$, we can now evaluate sentences:

- $\text{Teaches}(\text{Alice}, \text{CS101})$ says Alice teaches CS101. $\langle \text{Alice}, \text{CS101} \rangle \in \text{Teaches}$, so this is $\text{true}$.
- $\text{Teaches}(\text{Carol}, \text{CS101})$ says Carol teaches CS101. $\langle \text{Carol}, \text{CS101} \rangle \notin \text{Teaches}$, so this is $\text{false}$.
- $\text{EnrolledIn}(\text{Carol}, \text{CS101}) \land \text{Prerequisite}(\text{CS101}, \text{CS201})$. Both tuples are in their respective relations, so this conjunction is $\text{true}$.

A different model $m'$ with the same domain but $\text{Teaches} = \{ \langle \text{Carol}, \text{CS101} \rangle \}$ would make $\text{Teaches}(\text{Carol}, \text{CS101})$ true and $\text{Teaches}(\text{Alice}, \text{CS101})$ false— a different possible world, even though the objects are the same.

**A predicate is the name we use in a sentence to refer to a relation.** It is a symbol that takes one or more objects as arguments and evaluates to $\text{true}$ or $\text{false}$ depending on whether the corresponding tuple is in the relation.

In the university example above, $\text{Professor()}$, $\text{Student()}$, $\text{Teaches()}$, etc. are all predicates. When we write $\text{Teaches}(\text{Alice}, \text{CS101})$, we are applying the predicate $\text{Teaches}$ to the arguments $\text{Alice}$ and $\text{CS101}$. The model $m$ then determines whether that sentence is true by checking whether $\langle \text{Alice}, \text{CS101} \rangle$ belongs to the relation $\text{Teaches}$ in $m$.

The key distinction: the **relation** is the mathematical object (a set of tuples) that lives inside the model; the **predicate** is the syntactic symbol in the language that we use to talk about it. Writing $\text{Teaches}(\text{Alice}, \text{CS101})$ in a sentence is syntax; looking up whether $\langle \text{Alice}, \text{CS101} \rangle \in \text{Teaches}$ in the model to get a truth value is semantics.

### Quantifiers

Once we have a logic that allows objects, it is only natural to want to express properties of entire collections of objects, instead of enumerating the objects by name. Quantifiers let us do this. First-order logic contains two standard quantifiers, called **universal** and **existential**.

#### Universal quantification ($\forall$)

The symbol $\forall$ means "for all." A sentence $\forall x\; \alpha(x)$ is true in a model if and only if $\alpha(x)$ is true for every object in the domain when $x$ is substituted with that object.

**Example:** In model $m$, the domain is $\{ \text{Alice}, \text{Bob}, \text{Carol}, \text{CS101}, \text{CS201} \}$ and $\text{Professor} = \{ \langle \text{Alice} \rangle, \langle \text{Bob} \rangle \}$.

Consider the sentence:

$$\forall x\; \text{Professor}(x) \Rightarrow \text{Teaches}(x, \text{CS101})$$

"Every professor teaches CS101." We check every object in the domain:

- $x = \text{Alice}$: $\text{Professor}(\text{Alice})$ is true, and $\langle \text{Alice}, \text{CS101} \rangle \in \text{Teaches}$, so the implication holds. ✓
- $x = \text{Bob}$: $\text{Professor}(\text{Bob})$ is true, but $\langle \text{Bob}, \text{CS101} \rangle \notin \text{Teaches}$ (Bob teaches CS201). The implication fails. ✗
- $x = \text{Carol}, \text{CS101}, \text{CS201}$: $\text{Professor}$ is false for all three, so the implication is vacuously true. ✓

Since $x = \text{Bob}$ fails, the sentence is **false** in $m$.

#### Existential quantification ($\exists$)

The symbol $\exists$ means "there exists." A sentence $\exists x\; \alpha(x)$ is true in a model if and only if $\alpha(x)$ is true for at least one object in the domain.

**Example:** Using the same university domain:

$$\exists x\; \text{Student}(x)$$

This is true if at least one object in the domain satisfies $\text{Student}(x)$. In model $m$, Carol is a student, so the sentence is true. If we considered a model with no students in the domain, it would be false.

A combined example:

$$\exists x\; \text{Professor}(x) \land \text{Teaches}(x, \text{CS101})$$

This says "there exists someone who is a professor and teaches CS101." In model $m$, Alice satisfies this. So the sentence is true.

**Example:** Using the full model $m$, we can write and evaluate several quantified sentences.

**1.** $\forall x\; \text{Professor}(x) \Rightarrow \exists y\; \text{Course}(y) \land \text{Teaches}(x, y)$

"Every professor teaches at least one course." Without the $\text{Course}(y)$ guard, $y$ would range over all domain objects (including people), which is nonsensical. The predicate $\text{Course}$ is defined in $m$ and restricts $y$ to actual courses.

- For $x = \text{Alice}$: $\text{Professor}(\text{Alice})$ is true. Is there a course $y$ such that $\text{Teaches}(\text{Alice}, y)$? Yes. $\langle \text{Alice}, \text{CS101} \rangle \in \text{Teaches}$ and $\text{Course}(\text{CS101})$ holds. ✓
- For $x = \text{Bob}$: $\text{Professor}(\text{Bob})$ is true. $\langle \text{Bob}, \text{CS201} \rangle \in \text{Teaches}$ and $\text{Course}(\text{CS201})$ holds. ✓
- $x = \text{Carol}, \text{CS101}, \text{CS201}$: $\text{Professor}$ is false for all three, so the implication is vacuously true. ✓

The sentence is **true** in $m$.

**2.** $\exists x\; \text{Student}(x) \land \forall y\; \text{Course}(y) \Rightarrow \text{EnrolledIn}(x, y)$

"There exists a student who is enrolled in every course." Without the $\text{Course}(y)$ guard, $\forall y$ would range over people and courses alike (demanding, for example, that Carol be enrolled in Alice), which makes no sense.

- The only student is Carol. We check every object $y$ where $\text{Course}(y)$ holds— CS101 and CS201. Is $\langle \text{Carol}, \text{CS101} \rangle \in \text{EnrolledIn}$? ✓. Is $\langle \text{Carol}, \text{CS201} \rangle \in \text{EnrolledIn}$? ✓. For all other $y$ (Alice, Bob, Carol), $\text{Course}(y)$ is false, so the implication is vacuously true.

The sentence is **true** in $m$.

**3.** $\forall x\; \forall y\; \text{Teaches}(x, y) \Rightarrow \text{Professor}(x)$

"Only professors teach courses."

- Check every $\langle x, y \rangle \in \text{Teaches}$: $\langle \text{Alice}, \text{CS101} \rangle$. Is Alice a professor? Yes. $\langle \text{Bob}, \text{CS201} \rangle$. Is Bob a professor? Yes.

The sentence is **true** in $m$.

#### Connections between $\forall$ and $\exists$

The two quantifiers are interdefinable via negation— you only ever need one of them. This is the quantifier analogue of De Morgan's laws.

**De Morgan's laws for quantifiers:**

$$\lnot \forall x\; \alpha(x) \equiv \exists x\; \lnot \alpha(x)$$

$$\lnot \exists x\; \alpha(x) \equiv \forall x\; \lnot \alpha(x)$$

The first says: "it is not the case that all $x$ satisfy $\alpha$" is the same as "there exists some $x$ that does not satisfy $\alpha$." The second says: "there is no $x$ satisfying $\alpha$" is the same as "every $x$ fails to satisfy $\alpha$."

**Example using model $m$:**

Consider the sentence $\forall x\; \text{Professor}(x) \Rightarrow \text{Teaches}(x, \text{CS101})$. "Every professor teaches CS101." We showed earlier this is **false** in $m$ (Bob doesn't teach CS101). By the first law, its negation must be true:

$$\lnot(\forall x\; \text{Professor}(x) \Rightarrow \text{Teaches}(x, \text{CS101})) \equiv \exists x\; \lnot(\text{Professor}(x) \Rightarrow \text{Teaches}(x, \text{CS101}))$$

Recall that $\lnot(P \Rightarrow Q) \equiv P \land \lnot Q$, so this simplifies to:

$$\exists x\; \text{Professor}(x) \land \lnot\text{Teaches}(x, \text{CS101})$$

"There exists a professor who does not teach CS101." In $m$, Bob is that professor. ✓ The two sentences are equivalent. One is the negation of the other. Checking one is enough to know the truth value of both.

### First-Order Logic in AI

John McCarthy (1958) was primarily responsible for the introduction of first-order logic as a tool for building AI systems. The prospects for logic-based AI were advanced significantly by Robinson’s (1965) development of resolution, a complete procedure for first-order inference. The logicist approach took root at Stanford University. Cordell Green (1969a, 1969b) developed a first-order reasoning system, QA3, leading to the first attempts to build a logical robot at SRI (Fikes and Nilsson, 1971). The field of First-Order Logic took off from there.

One of the most prominent applications of FOL in AI was **expert systems**— programs designed to replicate the decision-making of a human expert in a narrow domain. An expert system encodes domain knowledge as a set of first-order rules (often written as Horn clauses of the form $P_1 \wedge P_2 \wedge \cdots \wedge P_n \Rightarrow Q$) and applies an inference engine to derive conclusions from those rules given observed facts.

Classic expert systems include MYCIN (medical diagnosis of blood infections), DENDRAL (chemical structure identification), and R1/XCON (computer configuration). These systems demonstrated that FOL-style reasoning could achieve expert-level performance in specific domains when the knowledge was carefully encoded. The rules in MYCIN, for example, were of the form: "if the infection is primary-bacteremia and the site is one of the sterile sites and the suspected portal of entry is the gastrointestinal tract, then there is evidence that the organism is bacteroides." This is essentially a quantified conditional in FOL.

Despite their successes, expert systems exposed a fundamental limitation of the hand-coded knowledge approach: the **knowledge acquisition bottleneck**. Encoding all the relevant facts and rules for even a moderately complex domain required enormous effort from human experts. Worse, a rule-based system struggles with genuinely novel situations. Anything not already covered by its rules falls through the cracks. Stuart Russell describes this problem directly in the context of self-driving cars: Google’s early architecture used "a 1970s-style rule-based expert system" to decide what to do, but "almost every day brought a situation the rules did not cover." Adding more rules never converged. See [Stuart Russell on the Long-Term Future of AI](../notes_from_ai_podcasts/dec_2018_stuart_russell_long_term_future_of_artificial_intelligence_lex_fridman_podcast_no_9.md).

Yann LeCun similarly characterizes expert systems and knowledge graphs as "too rigid and too brittle," pointing to the knowledge acquisition problem as essentially impractical at scale. See [Yann LeCun on Deep Learning](../notes_from_ai_podcasts/aug_2019_yann_lecun_deep_learning_convnets_and_self_supervised_learning_lex_fridman_podcast_no_36.md).

Yet Peter Norvig argues that expert systems got some things right: representation and reasoning remain crucial, especially when there is not enough data to learn from scratch. The insight that you need a way to represent structured knowledge and take steps of inference over it, is exactly what FOL provides, and it continues to underpin areas like knowledge graphs, etc. today. See [Peter Norvig on AI: A Modern Approach](../notes_from_ai_podcasts/sep_2019_peter_norvig_artificial_intelligence_a_modern_approach_lex_fridman_podcast_no_42.md).