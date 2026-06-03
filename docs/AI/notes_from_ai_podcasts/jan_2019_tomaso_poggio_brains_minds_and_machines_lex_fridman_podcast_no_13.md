# Jan 2019. [Tomaso Poggio: Brains, Minds, and Machines | Lex Fridman Podcast no. 13](https://www.youtube.com/watch?v=aSyZvBrPAyk&list=PLypW8HeibkIJMB_Yl3S5KAr69VBF3OPIa)
19 Jan 2019.

## Intelligence without understanding the brain

A central question is how far **intelligent systems** can be built without a functional understanding of how the **human brain** produces intelligence.

The **birds-and-planes** analogy is instructive but limited. Flight was achieved without deep biological theory, though it helped to know that heavy objects could fly. The Wright brothers observed birds, yet biology was not central to the engineering solution. 
Whether machines will match human capability *without* brain science is an **educated bet** that researchers with different backgrounds answer differently.

Many agree that, sooner or later, a machine could become **indistinguishable from a competent human assistant** in what it can be asked to do. The open question is whether that endpoint is reached with little neuroscience input or whether understanding the brain is the best path.

## Neuroscience and recent AI breakthroughs

Recent progress in AI (roughly the prior five to ten years) suggests that major breakthroughs often **start from neuroscience**:

- **[Reinforcement learning](dec_2018_pieter_abbeel_deep_reinforcement_learning_lex_fridman_podcast_no_10.md)** (RL) sits at the core of **AlphaGo**, which defeated Lee Sedol (see also [solved games and chess engines](../primer_on_ai/introduction_to_artificial_intelligence.md#solved-games-and-chess-engines) for why exact minimax is infeasible at Go scale). RL traces to Pavlov, Skinner, Marvin Minsky in the 1960s, and later neuroscientists.
- **Deep learning** is likewise central to AlphaGo and to autonomous driving (e.g. **Mobileye**, founded by former student Amnon Shashua). Layered architectures echo **Hubel and Wiesel** at Harvard in the 1960s.

Poggio’s view is that neuroscience will likely **continue to inspire** at least some future breakthroughs, not necessarily all of them, but enough to matter.

## Biological vs artificial neural networks

Artificial neural networks are a gross simplification, yet **architecturally closer to the brain** than earlier AI formalisms (**Lisp**, **Prolog**, mathematical logic).

A major gap today is in **labeled data**. Deep learning often needs huge human-labeled sets (e.g. **ImageNet** on the order of $10^6$ images). A child won't receive millions of explicit labels from caregivers.

In **Go**, the world is simple enough that **self-play** (as in AlphaZero-style training; see [Pieter Abbeel on RL and self-play](dec_2018_pieter_abbeel_deep_reinforcement_learning_lex_fridman_podcast_no_10.md#self-play)) can supply learning signal without human labels. The **visual world** is far harder.

## Nature, nurture, and evolution

The classic **nature vs nurture** question applies: how much is in the **genes** vs individual **experience**? Both matter.

Evolution, on a Darwinian view, is **opportunistic**. Human DNA does not have vastly more genes than *Drosophila*, yet humans learn richly during life while the fruit fly is largely (~95%) **hard-coded** by genes. A plausible evolutionary strategy is a **general learning machine** with **weak priors** rather than encoding every circuit in the genome.

## Face recognition and plasticity

Neurons in face-selective brain areas appear involved in **face recognition**; such an area exists in young children and adults. Open questions: is it **innate** or **learned very fast** (e.g. mother’s face)?

Poggio’s **hunch** was rapid learning. **Margaret Livingstone** (Harvard) raised infant monkeys **deprived of faces** early in life (caregivers wore masks). The usual face-selective region showed **no face preference**— suggesting a **plastic territory** predisposed to imprint easily, with genes specifying something like “memorize what you see often in the first weeks, especially with food,” not a full face template (which would cost many bits of wiring).

In that paradigm, monkeys often saw technicians’ **blue gloves** during feeding; some cells became **hand- or glove-sensitive** (instead of face-sensitive), imprinting on frequent early visual patterns tied to reward.

## Brain modularity

Mid-20th-century neuroscience briefly favored **equipotentiality**: any cortical region might substitute for another (Lashley’s lesion studies in rodents). That view failed. The brain has **specific modules**. Stroke in one region can abolish speech, another leg control, while retaining **plasticity** and partial **remapping**.

Evidence comes from **lesions** (animals and war injuries), and more recently **fMRI** and related imaging showing task-specific activation (e.g. language areas).

This connects to the broader [modularity](dec_2018_pieter_abbeel_deep_reinforcement_learning_lex_fridman_podcast_no_10.md#modularity) point in the Pieter Abbeel notes: capability may come from reusable modules rather than one undifferentiated system.

## Levels of understanding: brain vs computer

**Levels of abstraction** are all needed for the brain, as for any complex system.

For a **computer** (engineered in layers):

- Using **PowerPoint** is one level of “understanding.”
- Understanding **transistors**, diodes, and logic gates is another.
- Those levels are **deliberately separate**; chip designers need not know PowerPoint internals.

For the **brain**, **algorithms** (the computational level) and **circuits** (implementation) are **far more intertwined** than in engineered computers. Software and hardware are not cleanly isolated. Progress likely requires **collaboration across disciplines**. Poggio ranks understanding the brain among the **hardest problems in science**.

## Universal approximation and curse of dimensionality

The **universal approximation theorem** (finite neurons, one hidden layer can approximate broad function classes) was not surprising to Poggio. It parallels **Weierstrass Approximation Theorem** (every continuous function on a closed, bounded interval can be uniformly approximated to any desired degree of accuracy by a polynomial, provided you use enough terms).

The interesting question is **sample complexity**. What should be the dimensionality of our input to achieve a given error rate? In other words, what should be the dimensionality of our feature space for a given error rate? For error $\le 10\%$, shallow nets may need on the order of $10^{d}$ units for dimension $d$. A $200 \times 200$ image implies astronomical $d$. This brings in a lot of problems, including the [**curse of dimensionality**](nov_2025_mindscape_336_anil_ananthaswamy_on_the_mathematics_of_neural_nets_and_ai.md#the-curse-of-dimensionality).

Hope: **depth** and architectures with **local connectivity** (e.g. convolution) avoid the curse when the target function is **hierarchical/compositional**.

## Scene understanding

The gap from **object detection** to **scene understanding** (what is happening, language about actions, etc.) remains **large**. The current era is a **golden age for low-level vision and speech** (Alexa, medical imaging aids, etc.), but **understanding** scenes, language, and human action, despite hype, is still far off.

## Explainability and engineering AGI

Will AGI be **simple**, **explainable**, **engineered from first principles** like transistors → PowerPoint?

Poggio doubts that AGI will be **interpretable neuron by neuron** (or weight by weight). The same split already shows up in today’s models. A single unit in a **deep net** or **kernel machine** is usually opaque, but the **network as a whole** (what it computes, what it can and cannot do) is increasingly understood at the systems level.

## Ethics and the neuroscience of ethics

**Ethics** is likely **learnable**. Two directions: **ethics of neuroscience** (how researchers should behave) vs **neuroscience of ethics** (how moral judgment is implemented).

The latter is central for **ethical machines**. Evidence includes **fMRI** of regions involved in moral judgment and **TMS** (magnetic stimulation) altering decisions (work by **Rebecca Saxe** and others). If ethics has neural substrates, designing aligned systems is partly a **learnable** engineering problem.
