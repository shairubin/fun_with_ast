# Fun with AST


The Fun with AST library is driven by the aspiration to enhance the productivity and quality of a software engineer's
work. Its main goal is to enable developers to focus on implementing the core business values needed to be achieved,
rather than wasting time on repetitive and routine tasks that cab be done automatically. 

## A Hybrid Programming Model


You can see this idea in Figure 1. You, a software engineer
write the business code (1) and provide instructions (in the form of declarative configuration) (2) on the nature of code they
want to add in a consistent and repetitive way. The Fun with AST engine (3) is then automatically adds the necessary code to
the your original source code, producing code that contains both the business value and the additional code of the
repetitive task.

<p align="center" width="100%">
Figure 1: A hybrid programming model <br>
<img src="https://drive.google.com/uc?id=143ris5WmBqpzB52NH9NYaHxeHWgPF00-" 
width="60%"  alt="Alt text" title="Fun with AST concepts">
</p>

### Example of repetitive,  trivial or automated tasks: 

## Challenges 
## Using the library

See the [test-fun-with-ast](https://github.com/shairubin/test-fun-with-ast) project for examples of using the
fun-with-ast library.

## Why Fun-with-AST

1. Here is
   a [talk](https://docs.google.com/presentation/d/e/2PACX-1vQTQQNaUPs7UNO_skE5vxBxaYbu6box99g_DnYYOuXuIKUqxI-_XEMxQ3p0_CBNlE6V9F3NzpOaXzUJ/pub?start=true&loop=false&delayms=30000)
   I gave in Pycon 2023. It explains the capabilities of fun-with-ast.
2. It is a great learning personal development experience.
3. Enables smart and complex manipulations

## Examples: AST Parse vs. AST Unparse vs. fun-with-ast Source Code Preserver

The examples below show an original program that first was unparsed with
python ast module,
and then was unparsed using the fun-with-ast library. The actual code that generates
these example can be found in the [test-fun-with-ast](https://github.com/shairubin/test-fun-with-ast)
library.

1. [Parse-Unparse Challenge Examples](https://shairubin.github.io/fun_with_ast/docs/exampels.html)

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