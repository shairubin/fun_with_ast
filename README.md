# Fun with ASTs
Provides developers with a programmatic tool to change our own code.
This repository contains a library to analyze and manipulate python code using [Abstract Systax Tress](https://docs.python.org/3/library/ast.html) manipulation. 

## Using the library
See the [test-fun-with-ast](https://github.com/shairubin/test-fun-with-ast) project for examples of using the fun-with-ast library.


## Why Fun with ASTs
1. The intellectual problem: 
2. it is fun
2. it is a great learning experience 
3. enables smart and complex manipulations 

## How it works

## AST Parse and Unparse Examples
```html
<head>
	<meta http-equiv="content-type" content="text/html; charset=utf-8"/>
	<title></title>
	<meta name="generator" content="LibreOffice 7.3.7.2 (Linux)"/>
	<meta name="created" content="00:00:00"/>
	<meta name="changed" content="2023-07-01T21:17:00.500233146"/>
	<style type="text/css">
		@page { size: 21cm 29.7cm; margin: 2cm }
		p { line-height: 115%; margin-bottom: 0.25cm; background: transparent }
		pre { background: transparent }
		pre.western { font-family: "Liberation Mono", monospace; font-size: 10pt }
		pre.cjk { font-family: "Noto Sans Mono CJK SC", monospace; font-size: 10pt }
		pre.ctl { font-family: "Liberation Mono", monospace; font-size: 10pt }
	</style>
</head>
<body lang="en-IL" link="#000080" vlink="#800000" dir="ltr"><pre class="western">
<font color="#c9211e"><font size="3" style="font-size: 12pt"><u>ORIGINAL PROGRAM:</u></font></font>
<font color="#c9211e"><font size="3" style="font-size: 12pt"># A utility function that returns true if x is perfect square</font></font>
<font color="#c9211e"><font size="3" style="font-size: 12pt">import math</font></font>
<font color="#c9211e"><font size="3" style="font-size: 12pt">import logging</font></font>

<font color="#c9211e"><font size="3" style="font-size: 12pt">def isPerfectSquare(x):</font></font>
<font color="#c9211e">    <font size="3" style="font-size: 12pt">s = int(math.sqrt(x))</font></font>
<font color="#c9211e">    <font size="3" style="font-size: 12pt">return s*s == x</font></font>


<font color="#c9211e"><font size="3" style="font-size: 12pt"># Returns true if n is a Fibonacci Number, else false</font></font>

<font color="#c9211e"><font size="3" style="font-size: 12pt">def isFibonacci(n):</font></font>
<font color="#c9211e">    <font size="3" style="font-size: 12pt"># n is Fibonacci if one of 5*n*n + 4 or 5*n*n - 4 or both</font></font>
<font color="#c9211e">    <font size="3" style="font-size: 12pt"># is a perfect square</font></font>
<font color="#c9211e">    <font size="3" style="font-size: 12pt">return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)</font></font>
<font color="#c9211e"><font size="3" style="font-size: 12pt">#</font></font>
<font color="#c9211e"><font size="3" style="font-size: 12pt">#</font></font>
<font color="#c9211e"><font size="3" style="font-size: 12pt"># # A utility function to test above functions</font></font>
<font color="#c9211e"><font size="3" style="font-size: 12pt">for i in range(1, 15):</font></font>
<font color="#c9211e">    <font size="3" style="font-size: 12pt">if isFibonacci(i) == True:</font></font>
<font color="#c9211e">        <font size="3" style="font-size: 12pt">print('fun with ast')</font></font>
<font color="#c9211e">        <font size="3" style="font-size: 12pt">print(i, 'is a Fibonacci Number')</font></font>
<font color="#c9211e">    <font size="3" style="font-size: 12pt">else:</font></font>
<font color="#c9211e">        <font size="3" style="font-size: 12pt">print(i, 'is a not Fibonacci Number')</font></font>


<font color="#000000"><font size="3" style="font-size: 12pt"><u>AST UNPARSED PROGRAM:</u></font></font>
<font color="#000000"><font size="3" style="font-size: 12pt">import math</font></font>
<font color="#000000"><font size="3" style="font-size: 12pt">import logging</font></font>

<font color="#000000"><font size="3" style="font-size: 12pt">def isPerfectSquare(x):</font></font>
<font color="#000000">    <font size="3" style="font-size: 12pt">s = int(math.sqrt(x))</font></font>
<font color="#000000">    <font size="3" style="font-size: 12pt">return s * s == x</font></font>

<font color="#000000"><font size="3" style="font-size: 12pt">def isFibonacci(n):</font></font>
<font color="#000000">    <font size="3" style="font-size: 12pt">return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)</font></font>
<font color="#000000"><font size="3" style="font-size: 12pt">for i in range(1, 15):</font></font>
<font color="#000000">    <font size="3" style="font-size: 12pt">if isFibonacci(i) == True:</font></font>
<font color="#000000">        <font size="3" style="font-size: 12pt">print('fun with ast')</font></font>
<font color="#000000">        <font size="3" style="font-size: 12pt">print(i, 'is a Fibonacci Number')</font></font>
<font color="#000000">    <font size="3" style="font-size: 12pt">else:</font></font>
<font color="#000000">        <font size="3" style="font-size: 12pt">print(i, 'is a not Fibonacci Number')</font></font>


<font color="#00a933"><font size="3" style="font-size: 12pt"><u>FUN WITH AST PROGRAM:</u></font></font>
<font color="#00a933"><font size="3" style="font-size: 12pt"># A utility function that returns true if x is perfect square</font></font>
<font color="#00a933"><font size="3" style="font-size: 12pt">import math</font></font>
<font color="#00a933"><font size="3" style="font-size: 12pt">import logging</font></font>

<font color="#00a933"><font size="3" style="font-size: 12pt">def isPerfectSquare(x):</font></font>
<font color="#00a933">    <font size="3" style="font-size: 12pt">s = int(math.sqrt(x))</font></font>
<font color="#00a933">    <font size="3" style="font-size: 12pt">return s*s == x</font></font>


<font color="#00a933"><font size="3" style="font-size: 12pt"># Returns true if n is a Fibonacci Number, else false</font></font>

<font color="#00a933"><font size="3" style="font-size: 12pt">def isFibonacci(n):</font></font>
<font color="#00a933">    <font size="3" style="font-size: 12pt"># n is Fibonacci if one of 5*n*n + 4 or 5*n*n - 4 or both</font></font>
<font color="#00a933">    <font size="3" style="font-size: 12pt"># is a perfect square</font></font>
<font color="#00a933">    <font size="3" style="font-size: 12pt">return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)</font></font>
<font color="#00a933"><font size="3" style="font-size: 12pt">#</font></font>
<font color="#00a933"><font size="3" style="font-size: 12pt">#</font></font>
<font color="#00a933"><font size="3" style="font-size: 12pt"># # A utility function to test above functions</font></font>
<font color="#00a933"><font size="3" style="font-size: 12pt">for i in range(1, 15):</font></font>
<font color="#00a933">    <font size="3" style="font-size: 12pt">if isFibonacci(i) == True:</font></font>
<font color="#00a933">        <font size="3" style="font-size: 12pt">print('fun with ast')</font></font>
<font color="#00a933">        <font size="3" style="font-size: 12pt">print(i, 'is a Fibonacci Number')</font></font>
<font color="#00a933">    <font size="3" style="font-size: 12pt">else:</font></font>
<font color="#00a933">        <font size="3" style="font-size: 12pt">print(i, 'is a not Fibonacci Number')</font></font></pre>
</body>
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