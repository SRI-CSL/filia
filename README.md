# Cross-language Static Analysis

This document describes approaches to developing static
analysis tools for scripting languages.  The goal is to
develop a framework suitable for analyzing code written in
multiple languages.

## Proposed Workflow

1. Parse Python into an AST using libraries in Pyre (written in OCaML).

2. Define an MLIR dialect with the Python language primitives.  This will
need to be defined in Ocaml for integration with Pyre and in tablegen for
use with the MLIR C++ library.

3. Write OCaml code to take the Pyre-generated Python AST and
emit MLIR in the textual format.

4. Explore building static analysis of MLIR to typecheck Python code.

Repeat this with Javascript using the OCaML
[Flow parser](https://opam.ocaml.org/packages/flow_parser/)
also developed by Facebook.  This will allow reusing the MLIR
generation or FFI code.

## Research Questions

What should the initial goal be?  Gradual typing?

How much commonality is there between the Python and Javascript
MLIR dialects?

## Market Questions

Who are other competitors in the multi-language analysis market?

* [Sonartype Qube](https://www.sonarqube.org/features/multi-languages/) is
  a commercial static analysis offering.

* [Sonatype Lift](https://www.sonatype.com/products/sonatype-lift) is a
  cloud-native, collaborative, code analysis platform built for developers.

Are there gradual typing tools that use a language-independent IR?

### Python Tools

* [Pyre](https://pyre-check.org/)

Pyre include things that may be relevant:

* [Typeshed](https://github.com/python/typeshed)

Typeshed contains external type annotations for the Python standard library and Python builtins, as well as third party packages as contributed by people external to those projects.

### Javascript Tools

* [Flow](https://flow.org/)

Flow is a static type checker for JavaScript.

* [Typescript](https://www.typescriptlang.org/)

TypeScript is a strongly typed programming language that builds on JavaScript, giving you better tooling at any scale.
Developed by Microsoft.

### Ruby Tools

* [Sorbet](https://sorbet.org/)

A gradual typechecker for Ruby.  Developed by Stripe.

## Engineering Questions

Should we write bindings for MLIR in a language better suited for
transformations/analysis than C++?

What existing capabilities within MLIR should we reuse?

How far can we actually get on this vision given the project
budget?

What existing dialects should we use for encoding Python/Javascript?


## Risks

The overhead of translating into a language-independent IR may lead
to slow performance.

* What are good benchmarks to measure the overhead associated with translating into MLIR?


