# Fun with ASTs
Provides developers with a programmatic tool to change our own code.
This repository contains a library to analyze and manipulate python code using [Abstract Systax Tress](https://docs.python.org/3/library/ast.html) manipulation. 

## Using the library
See the [test-fun-with-ast](https://github.com/shairubin/test-fun-with-ast) project for examples of using the fun-with-ast library.


## Why Fun with ASTs
1. The intellectual problem: 
2. It is fun
2. It is a great learning experience 
3. Enables smart and complex manipulations 


## AST Parse and Unparse Examples
The examples below show an original program that first was unparsed with 
python ast module, 
and then was unparsed using the fun-with-ast library. The actual code that generates 
these example can be found in the [test-fun-with-ast](https://github.com/shairubin/test-fun-with-ast) 
library. 

1. [Fibonacci calculator](./docs/parse_vs_unparse_vs_fwa.pdf)
## Potential usages:
- Fun #1: Keep source to source transformations
- Fun #2: add log
- Fun #3: switch else / if 
- Fun #4: mutation testing switch `<` into `<=`
- Fun #5: for to while 
- Fun #6: for loop into tail recursion 
- Fun #7: Add node AND comment





# Get started 