# Fun with ASTs
This repository contains a library to analyze and manipulate python [Abstract Systax Tress](TBD).


## Why Fun with ASTs
1. The intellectual problem: 
2. it is fun
2. it is a great learning experience 
3. enables smart and complex manipulations 

## How it works



## Comparing ast.parse and ast.unparse
| parse                | unparse                 | comments  |
|----------------------|-------------------------|-----------|
| `if (a<7) or (b>6):` | `if  a < 7  or  b > 6:` | ccomments |
| `a=7 # comment`      | `a = 7`                 |           |


```python
print(ast.unparse(ast.parse("""if True:\n  a=2\nelse:\n  if d==8:\n    c=7""")))
if True:
    a = 2
elif d == 8:
    c = 7
```    

```python
print(ast.unparse(ast.parse("""if True:\n  a=2\nelse:\n  if d==8:\n    c=7\n  else:\n    c =8""")))
if True:
    a = 2
elif d == 8:
    c = 7
else:
    c = 8
```
```python
code = """
if True:
   a=2
else:
  if d==8:
    c = 7
  elif d==9:
    c=8"""   
print(ast.unparse(ast.parse(code)))
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