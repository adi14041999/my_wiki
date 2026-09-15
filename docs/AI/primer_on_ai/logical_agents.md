# Logical Agents

## Knowledge-Based Agents

The problem-solving agents encountered in revious chapters ([search](solving_problems_by_searching.md), [constraint satisfaction](constraint_satisfaction_problems.md), and [adversarial games](adversarial_search_and_games.md)) know things, but only in a very limited, inflexible sense. They know what actions are available and what the result of performing a specific action from a specific state will be, but they don’t know general facts. A route-finding agent doesn’t know that it is impossible for a road to be a negative number of kilometers long. An 8-puzzle agent doesn’t know that two tiles cannot occupy the same space.

The central component of a knowledge-based agent is its **knowledge base (KB)**. A knowledge base is a set of **sentences** — "sentence" here is a technical term, related but not identical to sentences in English or other natural languages. Each sentence is expressed in a **knowledge representation language** and represents some assertion about the world. When a sentence is taken as given without being derived from other sentences, we call it an **axiom**.

There must be a way to add new sentences to the knowledge base and a way to query what is known. The standard names for these operations are **TELL** and **ASK**, respectively.

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

Each time the agent program is called, it does three things. First, it TELLs the knowledge base what it perceives. Second, it ASKs the knowledge base what action it should perform— in the process of answering this query, extensive reasoning may be done about the current state of the world, the outcomes of possible action sequences, and so on. Third, it TELLs the knowledge base which action was chosen, and returns the action so it can be executed.

The details of the representation language are hidden inside three helper functions that form the interface between sensors and actuators on one side, and the core representation and reasoning system on the other. `MAKE-PERCEPT-SENTENCE` constructs a sentence asserting that the agent perceived the given percept at the given time. `MAKE-ACTION-QUERY` constructs a sentence asking what action should be performed at the current time. `MAKE-ACTION-SENTENCE` constructs a sentence asserting that the chosen action was executed. The details of the inference mechanisms, in turn, are hidden inside `TELL` and `ASK`.

For example, an automated taxi might have the goal of taking a passenger from San Francisco to Marin County and might know that the Golden Gate Bridge is the only link between the two locations. We can then expect it to cross the Golden Gate Bridge, because it knows that doing so will achieve its goal. Notice that this analysis is independent of how the taxi works at the implementation level — it doesn't matter whether its geographical knowledge is stored as linked lists or pixel maps, or whether it reasons by manipulating strings of symbols in registers or by propagating noisy signals through a network of neurons.

The agent maintains a knowledge base, KB, which may initially contain some **background knowledge**.