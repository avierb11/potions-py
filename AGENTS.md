# AGENTS.md

## Purpose

This file defines the coding standards for all changes made in this repository. Follow these instructions unless a more specific `AGENTS.md` file exists in a subdirectory.

When modifying existing code, preserve established behavior and make the smallest change necessary to complete the task correctly. Do not perform unrelated refactoring.

## General Priorities

When making implementation decisions, prioritize the following, in order:

1. Correctness
2. Static type safety
3. Clear and maintainable public interfaces
4. Immutability and explicit data flow
5. Functional composition
6. Readability
7. Performance, unless performance is an explicit requirement

Do not sacrifice correctness or clarity for brevity.

## Static Typing

Use explicit types wherever the language permits them.

Type all practical locations, including:

- Function and method parameters
- Return values
- Local variables when the inferred type is not immediately obvious
- Class attributes
- Module-level variables and constants
- Collection element types
- Callback and callable signatures
- Generic type parameters
- Intermediate values when annotations improve static analysis
- Public interfaces and exported symbols

Prefer precise types over broad types.

- Avoid `Any`, `object`, untyped collections, dynamic casts, and equivalent escape hatches.
- Do not suppress type-checker errors unless there is no sound alternative.
- Do not use a cast merely to silence the type checker.
- Prefer typed domain objects, records, enums, protocols, and discriminated unions over loosely structured dictionaries or strings.
- Represent optionality explicitly.
- Narrow union types as early as possible.
- Preserve type information across function boundaries.
- Use generics when they express a real relationship between input and output types.
- Avoid introducing generic parameters that do not improve type safety.

Code should pass the repository's configured static type checker without errors. Type-checking failures must be fixed at their source rather than hidden with suppressions or unnecessarily broad annotations.

## Public APIs Only

Use public APIs exclusively.

Do not call or depend on:

- Private methods
- Private functions
- Internal modules
- Undocumented implementation details
- Names marked as internal
- Names whose naming convention indicates that they are private
- Unstable APIs that are not intended for external use

This restriction applies to both project code and third-party dependencies.

If a required operation appears to be available only through an internal API:

1. Look for an equivalent public API.
2. Restructure the implementation to use supported interfaces.
3. Add an appropriate public abstraction within this project when necessary.
4. Document the limitation if no supported solution exists.

Do not bypass public abstractions for convenience.

## Documentation

Add comprehensive docstrings to all public modules, classes, functions, methods, and other exported interfaces.

Docstrings should explain:

- What the component does
- Why it exists, when that is not obvious
- The meaning of each parameter
- The return value
- Relevant invariants
- Expected side effects
- Exceptions or errors that may be raised
- Important assumptions
- Units, coordinate systems, dimensions, or data conventions when applicable
- Non-obvious behavior and edge cases

Docstrings should explain the interface and behavior, not merely restate the implementation.

For complex internal functions, add docstrings even when they are not publicly exported. Use inline comments only when they explain reasoning, constraints, or non-obvious decisions. Do not add comments that simply narrate the code.

Keep documentation synchronized with implementation changes.

## Immutability and Assignment

Prefer immutable values and single assignment.

- Assign each variable once whenever practical.
- Prefer constructing a new value over mutating an existing one.
- Prefer immutable data structures when suitable alternatives exist.
- Do not reuse a variable name for values with different meanings or types.
- Avoid stateful accumulation when a transformation can express the same operation.
- Keep mutable state local, explicit, and narrowly scoped when it is unavoidable.
- Do not mutate function arguments unless mutation is an explicit, documented part of the interface.
- Avoid hidden mutation through shared objects or global state.
- Prefer immutable configuration and domain objects.

Mutation is acceptable when it is clearly justified by performance, interoperability, or the semantics of the problem. When mutation is necessary, isolate it behind a small, well-documented interface.

## Functional Programming

Prefer functional programming techniques when they improve clarity and type safety.

Favor:

- Pure functions
- Referentially transparent transformations
- Explicit inputs and outputs
- Function composition
- Declarative collection transformations
- Immutable data
- Small functions with a single responsibility
- Typed higher-order functions
- Separating computation from side effects

Avoid:

- Hidden state
- Global mutable variables
- Functions whose behavior depends on implicit context
- Unnecessary mutation
- Deeply stateful control flow
- Side effects mixed into core domain logic
- Large procedures that combine validation, transformation, I/O, and presentation

Keep side effects at the boundaries of the system. Core domain logic should be deterministic and testable without filesystem, network, database, user-interface, or process-level dependencies whenever possible.

Do not force a functional abstraction when a straightforward expression is clearer. Functional style should improve comprehension, not obscure the algorithm.

## Function and Interface Design

Functions should be small, focused, and explicit.

- Give each function one clear responsibility.
- Prefer explicit parameters over hidden dependencies.
- Prefer returning values over modifying external state.
- Use descriptive names that communicate intent.
- Avoid boolean parameters when an enum or dedicated type would better express the available modes.
- Avoid functions with many unrelated parameters.
- Introduce a typed parameter object when several values form a coherent concept.
- Validate inputs at system boundaries.
- Encode invariants in types whenever practical.
- Return structured, typed results rather than loosely related tuples or dictionaries.
- Make invalid states difficult or impossible to represent.

Do not add abstractions solely to reduce line count. Add an abstraction when it clarifies a domain concept, enforces an invariant, removes meaningful duplication, or creates a stable interface.

## Error Handling

Handle errors explicitly and at the appropriate abstraction level.

- Do not silently ignore failures.
- Do not catch overly broad exceptions unless re-raising after adding useful context.
- Catch the most specific error type available.
- Preserve the original cause when translating errors.
- Use typed error values or domain-specific exceptions where appropriate.
- Write actionable error messages that identify the failed operation and relevant context.
- Do not use sentinel values when the type system can represent failure explicitly.
- Do not use assertions for ordinary runtime validation of external input.

Errors should be handled where the code has enough context to respond meaningfully. Otherwise, allow them to propagate.

## Side Effects and Dependencies

Make dependencies and side effects explicit.

- Pass dependencies as arguments or constructor parameters.
- Avoid service locators and hidden global dependencies.
- Separate I/O from transformation logic.
- Wrap external systems behind small, typed interfaces.
- Keep filesystem, network, database, environment, clock, and randomness access at architectural boundaries.
- Do not introduce a new dependency when the existing language or project libraries provide an adequate solution.
- Use dependencies through their documented public interfaces only.

New dependencies must have a clear purpose and should be narrowly integrated.

## Testing

Add or update tests for every behavioral change.

Tests should cover:

- Normal behavior
- Boundary conditions
- Invalid inputs
- Relevant failure modes
- Previously reported regressions
- Type-sensitive behavior
- Important invariants

Prefer deterministic tests.

- Do not depend on wall-clock timing when a controllable clock can be injected.
- Do not use live network services in unit tests.
- Avoid shared mutable fixtures.
- Keep each test focused on one behavior.
- Use descriptive test names that state the expected behavior.
- Test public behavior rather than private implementation details.
- Do not call private or internal functions solely to make testing easier.

If code is difficult to test through its public interface, improve the design rather than exposing implementation details unnecessarily.

## Refactoring Existing Code

When editing existing code:

- Make the smallest complete change needed for the task.
- Preserve unrelated behavior.
- Preserve public interfaces unless the task explicitly requires changing them.
- Do not rename, reorganize, reformat, or rewrite unrelated code.
- Do not combine feature work with broad cleanup.
- Improve nearby typing or documentation only when necessary to implement the requested change correctly.
- Add regression tests before or alongside bug fixes when practical.

A localized inconsistency is preferable to an unrelated repository-wide refactor unless the task explicitly requests broader normalization.

## Compatibility

Respect the versions and platforms supported by the repository.

- Do not use language features newer than the configured minimum version.
- Do not change public behavior without documenting the change.
- Preserve serialization formats, command-line interfaces, configuration schemas, and external contracts unless the task requires otherwise.
- Avoid relying on platform-specific behavior unless the project explicitly targets that platform.
- Treat warnings and deprecations as issues to resolve through supported APIs, not as messages to suppress.

## Code Quality Checks

Before considering a change complete, run the repository's applicable:

- Formatter
- Linter
- Static type checker
- Unit tests
- Integration tests
- Documentation checks
- Build or compilation checks

Resolve failures caused by the change. Do not disable checks, weaken configurations, or add blanket suppressions merely to make the checks pass.

If a check cannot be run, state clearly which check was not run and why.

## Agent Behavior

Before changing code:

1. Read the relevant files and nearby tests.
2. Identify the public interfaces involved.
3. Inspect the repository's formatting, linting, typing, and testing configuration.
4. Follow existing project conventions unless they conflict with this file.
5. Determine the smallest change that fully addresses the task.

While changing code:

1. Preserve type information.
2. Prefer immutable transformations and single assignment.
3. Keep side effects isolated.
4. Use only public APIs.
5. Add or update docstrings.
6. Avoid unrelated changes.
7. Add focused tests for changed behavior.

After changing code:

1. Review the diff for accidental or unrelated edits.
2. Run formatting and linting checks.
3. Run the static type checker.
4. Run relevant tests.
5. Confirm that documentation still matches behavior.
6. Report any checks that could not be run.

## Prohibited Shortcuts

Do not:

- Use private or internal APIs.
- Introduce `Any` or an equivalent dynamic type without a documented, unavoidable reason.
- Add type-ignore comments as a first-line solution.
- Disable linting or type-checking rules to accommodate new code.
- Mutate shared state when an explicit value transformation is practical.
- Add mutable global state.
- Hide dependencies in global variables or implicit context.
- Catch and discard errors.
- Change unrelated code.
- Rewrite a working component solely to impose a preferred style.
- Add speculative abstractions for hypothetical future requirements.
- Duplicate logic when a clear public abstraction already exists.
- Remove documentation, tests, validation, or type annotations to simplify an implementation.

## Conflict Resolution

If instructions conflict, apply the following precedence:

1. The explicit user request
2. The nearest directory-specific `AGENTS.md`
3. This file
4. Established repository conventions
5. General language or framework conventions

Requirements involving correctness, safety, compatibility, and public contracts take precedence over stylistic preferences.

If complying with a request would require an unsupported internal API, unsafe type suppression, or a breaking public-interface change, explain the conflict and choose the safest supported implementation.

## Relevant online documentation URLS: 
- PyTorch: https://docs.pytorch.org/docs/2.13/index.html
- Potions directory: /home/andrew/Documents/Research/Projects/potions
