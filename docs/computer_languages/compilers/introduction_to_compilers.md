# Introduction to Compilers

A compiler is a program that takes in source code written in one language (called the source language) and returns source code written in another language (called the target language).

Below are some examples.

**GCC (GNU Compiler Collection)**— Compiles C/C++ source code to machine code (native binary) for a target architecture like x86 or ARM.

**Clang**— Compiles C, C++, and Objective-C to LLVM IR (intermediate representation)

**TypeScript Compiler (tsc)**— Compiles TypeScript to JavaScript. This is an example of a source-to-source compiler (also called a transpiler), where both the input and output are high-level languages.

**Babel**— Transpiles modern JavaScript (ES2015+) to older JavaScript compatible with legacy browsers.

**Kotlin compiler**— Compiles Kotlin to JVM bytecode, JavaScript, or anything else based on the target platform.

## Stages of a Compiler

Compilers typically work in a pipeline of stages, each transforming the source into a progressively lower-level representation.

**1. Lexing (Tokenization)**— The raw source text is broken into a flat list of tokens: keywords, identifiers, literals, operators, punctuation. Whitespace and comments are usually discarded here.

**2. Parsing**— The token stream is consumed by a parser that checks grammar and builds an Abstract Syntax Tree (AST). This is where syntax errors are caught. The AST captures the structure of the program (e.g., an `if` node with a condition child and two branch children).

**3. Semantic Analysis**— The AST is walked to enforce rules that go beyond syntax: type checking, variable resolution (is this name in scope?), and detecting things like using a variable before declaring it. Most type errors surface here.

**4. Intermediate Representation (IR) Generation**— The validated AST is lowered into an IR (a simplified, language-agnostic form that's easier to analyze and optimize). LLVM IR and JVM bytecode are well-known examples.

**5. Optimization**— The IR is transformed to be faster or smaller without changing behavior. Common optimizations include inlining small functions, eliminating dead code, and constant folding (replacing `2 + 3` with `5` at compile time).

**6. Code Generation**— The optimized IR is translated to the target language or machine code. For native compilers this means emitting assembly or binary instructions; for transpilers it means emitting source code in the target language.

**7. Linking (for native compilers)**— Separately compiled object files and libraries are combined into a single executable. The linker resolves references between files (e.g., a function defined in one file and called in another).

Not every compiler has all these stages— a simple transpiler might go straight from Semantic Analysis to output code. But this pipeline describes the full picture for a production-grade compiler.

Let's look at a few examples in detail.

### C/C++ with Clang

1. **Preprocessing**— The C preprocessor expands `#include`, `#define`, and other directives, producing a single translation unit.
2. **Lexing & Parsing**— Clang tokenizes the preprocessed source and builds an AST.
3. **Semantic Analysis**— Type checking, name resolution, and language-rule enforcement. Clang is known for producing detailed, precise error messages at this stage.
4. **LLVM IR Generation**— The AST is lowered to LLVM IR, a typed, platform-neutral intermediate representation.
5. **Optimization**— LLVM's optimization passes run on the IR (e.g., inlining, dead code elimination, loop unrolling).
6. **Code Generation**— LLVM translates the optimized IR to target-specific assembly.
7. **Assembling**— The assembler converts assembly to an object file (`.o`).
8. **Linking**— The linker combines object files and libraries into the final executable.

### C/C++ with GCC

1. **Preprocessing**— The C preprocessor expands `#include`, `#define`, and other directives, producing a single translation unit.
2. **Lexing & Parsing**— GCC tokenizes and parses the source into its internal AST representation (called GENERIC).
3. **Semantic Analysis**— Type checking and language-rule enforcement.
4. **GIMPLE IR Generation**— The AST is lowered to GIMPLE, GCC's simplified IR where complex expressions are broken into three-address form. GIMPLE is machine-independent. It's GCC's middle-end IR, designed to be a clean, simplified representation of the program logic without any knowledge of the target architecture. While GIMPLE itself is machine-independent, it's an internal GCC data structure. It's not a stable, serializable format you can dump and ship around like LLVM IR (which you can write to a .ll or .bc file and pass between machines). In practice, GIMPLE lives in memory during a GCC compilation session. It's not designed to be an interchange format. If you want a portable IR you, LLVM IR is the standard choice.
5. **Optimization**— GCC's middle-end passes optimize GIMPLE. GCC then lowers to RTL (Register Transfer Language) for further back-end optimization.
6. **Code Generation**— RTL is translated to target-specific assembly.
7. **Assembling**— `as` converts assembly to an object file.
8. **Linking**— `ld` links object files and libraries into the final executable.

### Java

1. **Lexing & Parsing**— `javac` tokenizes and parses `.java` source into an AST.
2. **Semantic Analysis**— Type checking, name resolution, and annotation processing.
3. **Bytecode Generation**— The AST is compiled to JVM bytecode and written to `.class` files. This bytecode is platform-neutral.
4. **Class Loading (runtime)**— The JVM (Java Virtual Machine) is a software runtime (a software runtime is the environment that a program needs in order to execute) that executes Java bytecode. he JVM is providing the execution engine, managing the heap, and running the garbage collector among many things. Rather than running on bare hardware, bytecode runs on top of the JVM, which translates it into native instructions for whatever machine it's running on. This is what gives Java its "write once, run anywhere" property. The same `.class` file runs on any platform that has a JVM. At the class loading stage, the JVM loads `.class` files on demand and verifies bytecode integrity before execution.
5. **Interpretation**— The JVM initially interprets bytecode directly.
6. **JIT Compilation**— The performance of interpretation is nowhere near good enough compared to when we have an object file being executed. For example, an optimizer can look at the whole program ahead of time and make smart decisions: reorder instructions, eliminate redundant computations, inline functions. An interpreter typically sees only one instruction at a time. Enter 'Just In Time Compilation'. The HotSpot JIT compiler (part of the JVM) identifies "hot" components of the code like loops. These hot parts are transformed to machine code during runtime.

### Python

1. **Lexing & Parsing**— The interpreter tokenizes the `.py` source and builds an AST.
2. **AST Optimization**— Minor constant folding and simplification on the AST.
3. **Bytecode Compilation**— The AST is compiled to CPython bytecode (`.pyc` files), a compact instruction set for the Python Virtual Machine (PVM).
4. **Interpretation**— CPython is the interpretor of the PVM. CPython executes bytecode instructions one at a time. CPython does not JIT compile by default, though PyPy and other implementations do.

### JavaScript (V8 engine)

1. **Parsing**— V8 is Google's JavaScript engine— the piece of software that takes JavaScript code and executes it. V8 parses JS source into an AST. It uses a fast "lazy parsing" strategy, fully parsing only what's immediately needed.
2. **Bytecode Generation (Ignition)**— The AST is compiled to bytecode by V8's Ignition interpreter, which begins executing it immediately.
3. **Profiling**— Ignition collects type feedback— which functions are hot, and what types flow through them.
4. **JIT Compilation (TurboFan)**— Hot functions are sent to TurboFan, V8's optimizer which uses the type feedback to emit highly optimized machine code.
5. **Deoptimization**— If a type assumption turns out to be wrong at runtime, TurboFan deoptimizes back to bytecode and re-profiles.

### Kotlin

1. **Lexing & Parsing**— The Kotlin compiler tokenizes and parses `.kt` source into an AST.
2. **Semantic Analysis**— Type inference, type checking, and name resolution (Kotlin's type system is more expressive than Java's, so this stage is substantial).
3. **Kotlin IR Generation**— The compiler lowers the AST to Kotlin's own IR.
4. **Back-end compilation** (one of three paths): **JVM target**— IR is lowered to JVM bytecode (`.class` files), fully interoperable with Java. **JavaScript target**— IR is transpiled to JavaScript source, usable in a browser or Node.js. **Native target (via LLVM)**— IR is translated to LLVM IR, then compiled to native machine code for platforms like macOS, Linux, iOS, or Windows. No JVM required.
