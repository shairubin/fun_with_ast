# Fun with ASTs
This repository contains a library to analyze and manipulate python [Abstract Systax Tress](TBD).


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


### example #2 
```python
import ast
code = """
if a<7 or b>9:
    pass"""  
print(ast.unparse(ast.parse(code)))
```
Output: 
```python 
if 6 < 7 or 6 > 9:
    pass
```
### example 3
```python
import ast
print(ast.unparse(ast.parse("""if True:\n  a=2\nelse:\n  if d==8:\n    c=7""")))
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