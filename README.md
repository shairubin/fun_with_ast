# Fun with AST

The Fun with AST library is driven by the
aspiration to enhance the productivity and quality of a software engineer's
work. Its main goal is to enable developers to focus on
implementing the core business values needed to be achieved,
rather than wasting time on
repetitive and routine tasks that can be done automatically.

## A Hybrid Programming Model with Source-to-Source Transformations

We envision a hybrid programming model that combines human and machine elements.
Figure 1 visualizes the concept of
hybrid programming.

(1) A developer writes the business code. Then, the developer also
provides instructions (e.g., declarative
configuration) regarding the nature of code they want to add
consistently and repetitively. The Fun with AST engine (3)
automatically adds the required source code to the original source code,
creating code that includes **both** the business value
and the additional code for the repetitive task (4).

<p align="center" width="100%" height="100%">
**Figure 1: A hybrid programming model** <br><br>
<img src="https://drive.google.com/uc?id=1vXPpQ_gIbCmBQUdQEYnf_JFVbILppGSF" 
width="70%"  alt="Alt text" title="Fun with AST concepts">
</p>

### Example of repetitive, trivial or automated tasks

When thinking of software engineering tasks, we can split them into two groups.
The first group requires the engineer to implement specific and typically
innovative capabilities. Examples might include developing a new computer game or creating
a new algorithm for financial trading.
In the second group we find, tasks that are
common to every software, regardless of the business domain.
We provide some examples of such common tasks below.

1. Removing common code from `if/else` blocks is a common optimization to
   increase code readability
   ([link](https://xp123.com/articles/refactoring-pull-common-code-conditional/)).
2. Other `if/else` optimizations for readability and simplicity are suggested
   [here](https://sourcery.ai/blog/explaining-refactorings-3/). For example, reducing
   redundancy of code in nested if statements."
3. Optimizing while loops into collection-based operations is suggested in this
   [article](https://martinfowler.com/articles/refactoring-pipelines.html).

4. Here is a more challenging problem, which represents the next level of
   control flow optimization: automatically convert a non-tail recursion
   into a [tail recursion](https://en.wikipedia.org/wiki/Tail_call).
   While a few [compilers](https://blog.knoldus.com/tail-recursion-in-java-8/) do that, engineers are typically not
   aware of this
   possibility. Having a tool for source-to-source transformations can help
   engineers understand this optimization and use it safely, without introducing errors.
5. Refactor long functions
6. testing - complete asserts , creating mocks for all dependencies

While we're not there yet, the fun-with-ast library enables easier code
analysis and transformation to identify when such optimizations are possible.

## Challenges of Source-to-Source Transformations

Source-to-source transformations aim to produce **live** code that
engineers can continue working with after the transformation is performed.
Such transformations must be executed without breaking
the code and should preserve all original source-code characteristics, like
indentation, comments, spacing, and readability based on standards like PEP8.

### Source-to-Source transformation example: AST Parse vs. AST Unparse

Python allows you to perform source-to-source transformations using the AST module. You can parse source code into an
Abstract Syntax Tree (AST), apply necessary modifications based on the 
desired transformation, and then unparse the tree
back into source code. However, the unparsing process doesn't retain 
essential elements from the original code,
including comments, spacing, indentation, quote types, line length, 
line breaks, and more. I'll provide some examples of
these issues below.

## Using the library

## Why Fun-with-AST

1. Here is
   a [talk](https://docs.google.com/presentation/d/e/2PACX-1vQTQQNaUPs7UNO_skE5vxBxaYbu6box99g_DnYYOuXuIKUqxI-_XEMxQ3p0_CBNlE6V9F3NzpOaXzUJ/pub?start=true&loop=false&delayms=30000)
   I gave in Pycon 2023. It explains the capabilities of fun-with-ast.
2. It is a great learning personal development experience.
3. Enables smart and complex manipulations

## Potential usages:

- Fun #1: Keep source to source transformations
- Fun #2: switch `else` / `if` bodies
- Fun #3: mutation testing switch `<` into `<=`
- Fun #4: Switch `For` to `While`
- Fun #5: for loop into tail recursion.
- Fun #7: Add node AND comment.

## How to Contribute

1. Follow the steps
   in  [Contribute to projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).
2. You can chose an existing open issue or open a new one.
3. Start working ....
4. Before submitting a pull request make sure tests are [passing](#how-to-run-tests).

## How to Run Tests

1. In `fun-with-ast` we use pytest.
2. Use your IDE to run all tests in `tests` directory.
3. OR, use command line:
    1. `cd <your path to fun-with-ast fork>/fun_with_ast/tests`
    2. `pytest --version`, should be at least `7.2.2`
    3. run `pytest`
    4. No tests should fail - some tests would be skipped / xfail.

## Limitations

1. The library is not yet mature.
2. Determining the the of quote (i.e., ' or ") for each string is done at the module or node level.
   If a node contains both `print('fun-with-ast')` and `print("fun-with-ast")` only one of them will be
   preserved,
   the first one in the module/node. (See method `guess_default_quote_for_node`, if you want to work on something
   interesting) 