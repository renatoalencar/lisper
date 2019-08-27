# Lisper

A very simple Lisper interpreter that supports only numbers.

## Introduction

I was there, on a sunday morning watching videos on YouTube when I get to [this Lecture by
Hal Abelson](https://youtu.be/-J_xL4IGhJA) on a introduction to Lisp. Is quite simple, it's almost an introduction to
programming itself and introduces the syntax and some operators of Lisp. It also introduces
the Haron of Alexandria algorithm of solving a square root of a number. So I thought: "Why
not build a Lisp interpreter that supports the exactly same specs he introduces on this
Lecture?".

And so I spent my sunday writing a very simple interpreter, and that here is.

## Running

The example in the lecture can be found in the file [sqrt.lisp](./sqrt.lisp), just pass the file name
as an argument like this:

```sh
$ ./lisper.py sqrt.lisp
1.4142135623746899
```

You can run `./lisper.py -h` to get some other options available like printing the AST or the
resulting value of each evaluation.

## Issues

This is a simple implementations, so here are a few known issues:

* It's not optimized;
* Only support arithmetic operations;
* There is no interop with Python;
* Does not keep track of lines and columns on errors.

## License

[MIT](./LICENSE)