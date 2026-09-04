# May 2019. [Chris Lattner- Compilers, LLVM, Swift, TPU, and ML Accelerators | Lex Fridman Podcast no. 21](https://www.youtube.com/watch?v=yCd3CzGSte8)
13 May 2019.

## What a compiler is

Lattner frames compilation as a two-sided problem. Humans need to write code, and machines need to run it. Humans do not want to write binary or think about every piece of hardware, and there are many kinds of hardware: x86, PowerPC, ARM, GPUs, and increasingly accelerators built for machine learning. On the other side there are many languages (C, JavaScript, Python and the rest) each trying to let a human express intent in a different way.

A **compiler** is a program that takes in source code written in one language (called the source language) and returns source code written in another language (called the target language) (see [Introduction to Compilers](../../computer_languages/compilers/introduction_to_compilers.md) for the general picture). A programming language's job is not only to talk to the compiler and the hardware; it is to capture an expression of what the programmer wanted, in a form that other humans can maintain, adapt and evolve.

Because both ends are complicated, compilers are built in phases, largely to maximize reuse of a very expensive body of code. Lattner collapses the usual [stages](../../computer_languages/compilers/introduction_to_compilers.md#stages-of-a-compiler) into three groups:

1. **Front end (parser):** language specific. Clang is the front end for C, C++ and Objective-C.
2. **Middle (optimizer):** largely language and hardware independent.
3. **Back end (code generation):** hardware specific.

## LLVM as shared infrastructure

[LLVM](https://llvm.org/) standardizes the middle and last parts. Swift, Julia, and Rust are very different languages, but they can all share the same optimization infrastructure and the same code generation and hardware support.

LLVM is an implementation rather than a specification— a body of code people reuse to build compilers. It is called a *compiler infrastructure* because it is the platform a concrete compiler is built on top of.

It is also a community of hundreds of people. One of the more remarkable things about LLVM is that harsh commercial competitors collaborate on it: Google and Apple, AMD and Intel, NVIDIA and others on the graphics side. They do this not out of altruism but because good shared infrastructure is in their commercial interest, and because building it all in-house is prohibitive— both in cost and in the scarcity of the skill set.

LLVM began as a university project at the University of Illinois. Lattner, his advisor, and two or three research students built many of the core pieces. He then went to Apple, which brought it into products first in the OpenGL graphics stack, then into the C compiler world, then Swift. As momentum built, others contributed. Google now effectively owns Clang because it cares so much about C++ and its tooling ecosystem, and NVIDIA's CUDA uses Clang and LLVM.

## Front ends are hard: Clang and C++

C++ is roughly 1400+ pages of specification. Complexity comes from syntax, from semantics, and from history. C++ builds on C, suboptimal decisions compound, and more things keep getting added. The interactions between subsystems are where the difficulty lives.

At the time, GCC was the industry-standard compiler, but it was hard to work with for research: difficult design, full of global variables, and not built for reuse outside its original purpose. Clang set out to do better on three fronts:

- **Better error messages** than GCC, which requires efficient bookkeeping.
- **Better compile time**, which is in tension with keeping the extra information good errors need.
- **New tools**— refactoring and analysis tools that GCC never supported, which then get built into IDEs.

That tooling contribution is arguably where Clang pushed the world forward most.

## Intermediate representations

After parsing, the front end produces an **abstract syntax tree**: a node for the `+` the human wrote, a node for a call with its function and arguments, and so on. Everything is nested.

That tree is then *lowered* into an [**intermediate representation**](../../computer_languages/compilers/introduction_to_compilers.md#intermediate-representation-ir). LLVM's IR is a **control flow graph**. Each operation is simple (add two numbers, multiply two things, make a call) and these are grouped into **blocks** of straight-line operations, with conditional branches between blocks.

A loop illustrates the difference. In a syntax tree, a `for` statement is one node with pointers to its initializer, comparison, increment and body. In a control flow graph you get a block for the initializer, a block for the body (with the increment in it), a comparison, a branch back to the top, and a branch out.

This is closer to an assembly-level representation, and that is the point: it is far more language independent. For example, JavaScript's particular ideas about what counts as false can stay in the front end, and the middle part can be shared.

## Classic optimizations

**Register allocation** was one of the big wins early on. Memory is relatively slow; registers are fast but scarce. An x86 machine in some modes has only eight. The compiler has to decide which values live in which registers at which points in the program. If an inner loop executing millions of times has to do loads and stores, it is slow; if its values fit in registers, it is fast.

**Scheduling** became important with RISC chips. Pipelines let a processor do more than one thing at a time, which makes the order of operations matter a great deal. The compiler moves instructions around so the pipelines stay full instead of stalling.

These techniques have been studied for decades, but the engineering of making them real is still hard.

## Machine learning for compilers

Many of these algorithms are full of hand-rolled heuristics and magic numbers that work well on specific benchmarks and do not generalize. Lattner sees this as an obvious opportunity for machine learning.

What you would optimize for is a choice: running time, memory use, or code size, which matters a lot in the embedded space.

## What actually changed the field

Progress in optimization has been largely incremental. The big jumps came from hardware and from language platforms.

**Java** changed the world. None of its individual ingredients were novel, but it pulled them together and made them mainstream: JIT compilation, garbage collection, portable code, memory-safe code, a dynamic dispatch execution model. JavaScript was another such shift.

The Java virtual machine also split the pipeline in a new place. Java parses to **bytecode**, an industry-standard, locked-down, portable representation. The optimizer and code generator can then be built by different vendors. Because the bytecode is memory safe and relatively trusted, it can be shipped across the wire.

Lattner thinks this was a good idea for the problems it solved— technology is neither good nor bad, it is how you apply it. Java did not win the desktop, but it has been very successful on servers for decades.

On the hardware side, **multi-core and vector instructions** did not remove any old problems but added new ones: how do you find enough work to keep a four-wide vector unit busy, how do you maximally use one core's arithmetic compute, and then how do you take that to multiple cores.

## LLVM's real contribution

Lattner does not point to compiler research innovations as LLVM's most profound outcome. It has good implementations of important algorithms, but what stands out is that **standardization made things possible that otherwise would not have happened**.

Sony, for example, picked up LLVM to do the graphics compilation in its movie production pipeline. That is not what it was designed for, which is the sign of good infrastructure. GCC is also great in various ways, but it is a C compiler or a Fortran compiler; it is not infrastructure in the same sense.

Modularity is the reason. Lattner wrote an early register allocator; someone else came along and replaced it, and could do so because the system was designed for that. In GCC, replacing a subsystem is possible but extremely difficult. This is also why LLVM did so well in the research world.

## Governance and community

LLVM uses a hierarchical system of **code owners**. Their responsibility is not to do all the work or review all the patches, but to make sure patches do get reviewed and that the right things happen architecturally in their area. Hardware manufacturers typically own the parts specific to their hardware. More often the process is organic: someone doing consistently good work becomes the de facto owner, and eventually somebody proposes making it official and everyone agrees.

Lattner is nominally still at the top of that stack, but spends his time negotiating technical disagreements and keeping the community moving in the right direction. The **LLVM Foundation**, a nonprofit, handles the business side (funding and running community events) and deliberately stays out of technical direction.

## Apple: Clang, then Swift

LLVM proved very useful to Apple through a series of transitions— Intel, 64-bit, the move to the iPhone. But the developer experience around Objective-C was not great: error messages, compile time, turnaround cycle, tooling and IDE.

Swift began just after the first version of C++ support in Clang was finished. C++ is formidable and important and also ugly in places, and you cannot work on it without thinking there has to be a better thing. Lattner started with no ambition that it would go anywhere, in his spare time, not telling anyone. It made good progress, and Bertrand Serlet, then senior VP of software, was encouraging and helped guide the early work.

The hard part was not technical. The prevailing view at Apple was that the iPhone was successful *because of* Objective-C. Apple hired software people who loved Objective-C (they did not come despite it) and the leadership lineage went back to NeXT, where Objective-C became real. Proposing a new language was heretical.

Lattner's sense was that the outside community was not in love with Objective-C. Some of the most outspoken people were, but many others were hitting its sharp corners and finding it difficult to learn.

The argument that made Swift happen was **memory safety**. The obvious counter-proposal was to file off Objective-C's rough edges instead. But Objective-C's object system is built on top of C pointers; pointers are unsafe, and if you remove the pointers it is not Objective-C anymore. Safety could not be fixed without fundamentally changing the language.

## Swift design choices

Some choices followed from context. A **typed language** was near-automatic: Objective-C is typed, and they wanted the performance and the refactoring and analysis tooling that types enable.

**Compiled rather than JIT** was less obvious historically. Apple seriously considered moving to Java in the late 90s. Being able to compile the code, ship it, and run standalone code that is not JIT compiled was a very big deal, and fits Apple's value system.

## TPUs and hardware/software co-design

### What a TPU is

A **TPU** (Tensor Processing Unit) is an ASIC (a chip built for one purpose) that Google designed specifically to run neural networks. The premise is that a general-purpose processor spends most of its transistors on things a neural network does not need. A CPU spends them on caches, branch prediction and out-of-order execution, all in service of running unpredictable, branchy code fast. A GPU spends them on a large number of programmable cores originally meant for graphics. But training and inference are dominated by one operation: dense matrix multiplication. A TPU spends nearly all of its transistors on that.

The centerpiece is the **MXU**, a matrix multiply unit built as a *systolic array*: a grid of multiply-accumulate cells, $128 \times 128$ in the v3 generation.

The generations track a widening ambition. The first TPU (2015) did inference only, in 8-bit integer arithmetic. The second added training, and with it bfloat16. The third, the generation Lattner is describing, was liquid cooled: each chip does roughly 123 teraflops in bfloat16 with 32 GiB of on-package memory, and a full **pod** of 1024 chips wired in a 2D torus reaches about 126 petaflops. That pod is what Lattner means by "100 petaflops in a large liquid-cooled box"— a room-scale machine that the software is meant to treat as a single device.

Crucially, you do not program a TPU the way you program a CPU. There is no meaningful sense in which a person hand-writes TPU assembly. You write a model, and a compiler (XLA in TensorFlow's case) turns the graph into code for the chip. This is why TPUs belong in a conversation about compilers at all: the hardware is only as good as the compiler that can target it.

### Co-design

Lattner describes TPUs as a perfect example of **hardware/software co-design**: deciding what hardware to build for certain classes of machine learning problems.

The showcase example is **bfloat16**, a compressed 16-bit floating point format that places its bits differently than standard fp16: a smaller mantissa and a larger exponent. It is less precise but represents a much larger range of values, which matters in machine learning, where you may need to accumulate very small gradients and also handle large-magnitude numbers. There are theories that the reduced precision actually helps generalization.

The area and time of a multiplier is quadratic in the number of mantissa bits but only linear in the size of the exponent.

bfloat16 came out of research (originally work on compressing weights for transport across a network) and then got burned into silicon.

## Open sourcing TensorFlow

Lattner thinks open sourcing TensorFlow changed the entire machine learning field and caused a revolution in its own right. It is easy to imagine a different world in which a company decided machine learning was too critical to share. That decision was very non-obvious at the time, and it has worked out well— for the field, and for Google.
