# Fun with ASTs
Provides developers with a programmatic tool to change our own code.
This repository contains a library to analyze and manipulate python code using [Abstract Systax Tress](https://docs.python.org/3/library/ast.html) manipulation. 

## Using the library
1. We assume you use python > 3.6 
2. Create a virtual environment: `python3 -m venv /path/to/new/virtual/environment`
3. Activate the virtual environment: `source /path/to/new/virtual/environment/bin/activate`
4. Install the library: `pip install fun-with-ast`
5. Run the library example programs:
    1. TBD
    2. TBD 
6. Deactivate the virtual environment: `deactivate`


## Why Fun with ASTs
1. The intellectual problem: 
2. it is fun
2. it is a great learning experience 
3. enables smart and complex manipulations 

## How it works

## AST Parse and Unparse Examples

### example #1: losing comments 
```python
import ast
code = """
a=7 # A is 7
"""  
print(ast.unparse(ast.parse(code)))
```
*Output:* 
```python
a = 7 
```
### example #2: Losing parentheses 
```python 
import ast
code ="""
if (a<7) or (b>9):
    pass
"""
print(ast.unparse(ast.parse(code)))
```
*Output:* 
```python
if a < 7 or b > 9:
    pass
```
### Example 3: Losing `elif`, losing indentation  
```python
import ast
code = """
if True:  
  a=2
else:
  if d==8:
    c=7
"""
print(ast.unparse(ast.parse(code)))
```
*output:*
```python
if True:
    a = 2
elif d == 8:
    c = 7
```    

```python
import ast
print(ast.unparse(ast.parse("""if True:\n  a=2\nelse:\n  if d==8:\n    c=7\n  else:\n    c =8""")))
if True:
    a = 2
elif d == 8:
    c = 7
else:
    c = 8
```
### example #1
```python
import ast
code = """
if True:
   a=2
else:
  if d==8:
    c = 7
  elif d==9:
    c=8"""   
print(ast.unparse(ast.parse(code)))
```
``python
if True:
    a = 2
elif d == 8:
    c = 7
elif d == 9:
    c = 8
```    
## examples
### Fun #1: Keep source to source transformations
### Fun #2: add log
### Fun #3: switch else / if 
### Fun #4: mutation testing switch `<` into `<=`
### Fun #5: for to while 
### Fun #6: for loop into tail recursion 
### Fun #7: Add node AND comment 
### Fun #8: Sort methods public to private






# Get started 