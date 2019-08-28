#!/usr/bin/env python

import sys
import json
import getopt

# Defines a set of valid non-alphanumeric characters
# for symbols. It's a constant in order to be a O(1)
# query evertime the lexer looks for it.
VALID_SYMBOL_CHARS = set('!-?+/*<>=')


def is_valid_symbol_char(c):
    '''
    Is this a non-alphanumeric character that is valid for
    symbols?

    :param c - a string of length 1 (a char)

    :return - a boolean
    '''
    return c in VALID_SYMBOL_CHARS


def is_valid_symbol(c):
    '''
    Is this character valid for symbols?

    :param c - a string of length 1 (a char)

    :return - a boolean
    '''
    return c.isalpha() or c.isdigit() or is_valid_symbol_char(c)


class Stream:
    '''
    Define basic stream definitions for the lexer (Tokenizer)
    and Parser.

    :param stream - Anything that can be indexed with continous integer value.
    '''
    def __init__(self, stream):
        self.stream = stream
        self.state = 0

    def finished(self):
        '''
        Did we finished the stream?

        :return - a bool
        '''
        return len(self.stream) <= self.state

    def current(self):
        '''
        Which is the current element in the stream?

        :return - a bool
        '''
        return self.stream[self.state]

    def consume(self, count):
        '''
        Go forward or backward (count < 0) in the stream.

        :param count - how many elements to skip (an int).
        '''
        self.state += count


class Token:
    '''
    Defines a token object

    :param type     - the token type (a str)
    :param value    - the token value (if present)
    :param **kwargs - token metadata
    '''
    def __init__(self, type, value=None, **kwargs):
        self.type = type
        self.value = value
        self.meta = kwargs
    
    def serialize(self):
        '''
        Serialize token object to a dict.
        '''
        return {
            'type': self.type,
            'value': self.value,
            'meta': self.meta
        }


class TokenList(list):
    def serialize(self):
        '''
        Serialize list of tokens
        '''
        return list(map(lambda token: token.serialize(), self))


class Tokenizer(Stream):
    '''
    Defines the language lexer
    '''

    def consume_symbol(self):
        '''
        Consumes and returns a symbol token

        :return - a str
        '''
        symbol = ''
        while is_valid_symbol(self.current()):
            symbol += self.current()
            self.consume(1)
        
        return symbol
    
    def consume_number(self):
        '''
        Consume a number token

        :return - a tuple (str, bool)
        :raises SyntaxError when there is not a valid number
        '''
        num = self.current()
        self.consume(1)
        is_int = True
        error = False

        while self.current().isdigit() or self.current() == '.':
            if self.current() == '.':
                if is_int:
                    is_int = False
                else:
                    error = True

            num += self.current()
            self.consume(1)

        if error:
            raise SyntaxError('not a valid number: %s' % num)
        
        return num, is_int
    
    def consume_comment(self):
        '''
        Consumes and ignore comments, comments starts with a semicolon
        and goes to the end of the line.
        '''
        while self.current() != '\n':
            self.consume(1)

    def tokenize(self):
        '''
        Consumes and generate token objects.

        :return - a list of Token objects.
        :raises SyntaxError when there's a unexpected character.
        '''
        tokens = TokenList()

        while not self.finished():
            c = self.current()
            if c == '(':
                tokens.append(Token('LPAREN'))
                self.consume(1)
            elif c == ')':
                tokens.append(Token('RPAREN'))
                self.consume(1)
            elif is_valid_symbol(c) and not c.isdigit():
                tokens.append(Token('SYMBOL', self.consume_symbol()))
            elif c.isdigit():
                num, is_int = self.consume_number()
                tokens.append(Token('NUMBER', num, int=is_int))
            elif c == ';':
                self.consume_comment()
            elif c.isspace():
                self.consume(1)
            else:
                raise SyntaxError('Unexpected character "%s"' % c)

        return tokens


class Node:
    '''
    Defines a node on the AST

    :param type - the node type
    :param value - the node value (if applicable)
    '''
    def __init__(self, type, value):
        self.type = type
        self.value = value
    
    def serialize(self):
        '''
        Serialize Node object to a dict.
        '''
        
        if self.type == 'expression':
            value = list(map(lambda i: i.serialize(), self.value))
        elif isinstance(self.value, Node):
            value = self.value.serialize()
        else:
            value = self.value

        return {
            'type': self.type,
            'value': value
        }
    
    def __str__(self):
        if self.type == 'literal':
            return self.value
        elif self.type == 'reference':
            return '<Node reference: %s>' % self.value
        
        return '<Node type: %s>' % self.type


class Program:
    '''
    The program definition (it's the AST itself here),
    which is basically a list of nodes.

    :param nodes - a list of Node objects.
    '''
    def __init__(self, nodes):
        self.nodes = nodes
    
    def serialize(self):
        '''
        Serialize program
        '''
        return list(map(lambda node: node.serialize(), self.nodes))


class Parser(Stream):
    '''
    Parses a token stream and generates an AST
    '''
    def expect(self, expected):
        '''
        Expects a Token of a type and raises an error if not found.

        :param expected - The expected type of the token

        :return - a bool (Always True, raises an error otherwise)
        :raises SyntaxError when the token stram end prematurely
                or the next token does not match the expected one.
        '''
        if self.finished():
            raise SyntaxError('Unexpected end of file')

        if self.current().type != expected:
            raise SyntaxError('Expected %s but received %s' % (expected, self.current()))
        
        return True
    
    def parse_number(self):
        '''
        Parse numbers

        :return - a tuple (type, value)
                  type - either 'int' or 'float'
                  value - the parsed value (float or int)
        '''
        if self.current().meta['int']:
            value = int(self.current().value)
            type = 'int'
        else:
            value = float(self.current().value)
            type = 'float'
        
        return Node('literal', value)

    def parse_expression(self):
        '''
        Parse an expression, also called a combination.

        :return - a list of parts of the expression.
        '''
        expression = []

        while True:
            if self.finished():
                raise SyntaxError('Unexpected end of expression')
            
            if self.current().type == 'SYMBOL':
                expression.append(Node('reference', self.current().value))
                self.consume(1)
            elif self.current().type == 'NUMBER':
                expression.append(self.parse_number())
                self.consume(1)
            elif self.current().type == 'RPAREN':
                self.consume(1)
                return Node('expression', expression)
            elif self.current().type == 'LPAREN':
                self.consume(1)
                expression.append(self.parse_expression())
            else:
                raise SyntaxError('Unexpected %s' % self.current())


    def parse(self):
        '''
        Parse the entire token stream and returns an AST.

        :return - an AST.
        '''
        nodes = []

        while not self.finished():
            self.expect('LPAREN')
            self.consume(1)
            expression = self.parse_expression()
            nodes.append(expression)

        return Program(nodes)


class Scope:
    '''
    Defines the runtime scope, you can add to other scope
    objects and dicts to get new ones and have local scopes.

    :param **kwargs - scope initial definition
    '''
    def __init__(self, **kwargs):
        self.state = kwargs

    def get(self, name):
        '''
        Get the value of something in the scope

        :param name - the name of the reference on the scope
        '''
        if name not in self.state:
            raise Exception('%s not in scope' % name)

        return self.state[name]

    def set(self, name, value):
        '''
        Set the value of something on the scope

        :param name - the name of the reference
        :param value - the value of that reference
        '''
        self.state[name] = value

    def __add__(self, y):
        '''
        Add new state the current scope and return a new one
        updated.

        :param y - new scope to be added.

        :return - a new scope
        '''
        x = self.state.copy()

        if isinstance(y, Scope):
            x.update(y.state)
        else:
            x.update(y)

        return Scope(**x)


class VM:
    '''
    The program runtime, which is responsible for evaluating the program
    too.

    :param scope - the initial scope of the runtime.
    '''
    def __init__(self, scope, print_i=False):
        self.scope = Scope(**scope)
        self.current_runtime_value = None
        self.print_i = print_i
    
    def build_scope(self, scope):
        '''
        Build a new scope merging with the current one

        :param scope - a Scope object or a dict

        :return - a new Scope object
        '''
        local = self.scope
        if scope is not None:
            local += scope
        
        return local

    def eval(self, node, scope=None):
        '''
        Eval something, it can be a Program or a Node object.

        :param node - an evaluable object (Program or Node).
        :param scope - the scope to evaluate the node on.

        :return - the value of the evaluated node.
        '''
        local = self.build_scope(scope)

        if isinstance(node, Program):
            for n in node.nodes:
                self.current_runtime_value = self.eval(n)
            
            value = self.current_runtime_value
        elif node.type == 'expression':
            value = self.eval_expression(node, local)
        elif node.type == 'literal':
            value = node.value
        elif node.type == 'reference':
            value = local.get(node.value)
        else:
            raise SyntaxError('unkown error:' + str(node))
        
        if self.print_i:
            print(value)

        return value
    
    def eval_expression(self, node, scope=None):
        '''
        Evaluate a Node of type expression
        '''
        expression = node.value
        f = self.scope.get(expression[0].value)
        args = []

        if not callable(f):
            raise RuntimeError('Operator must be a function')
        
        for i, arg in enumerate(expression[1:]):
            meta = f.meta['args'].get(i, {})
            
            if not meta.get('evaluate', True):
                args.append(arg)
            else:
                args.append(self.eval(arg, scope))
        
        return f(*args, scope=scope)


def meta(m=[]):
    '''
    Defines a function metadata to be used in the runtime
    '''
    def f(func):
        func.meta = {
            'args': dict(m)
        }
        return func
    return f


@meta([
    (0, { 'evaluate': False })
])
def define(name, value, scope):
    '''
    The `define` operator of Lisp. Set a value on the scope.

    (define <name> <value>)
    '''
    scope.set(name.value, value)
    return name


@meta()
def sum(*args, **kwargs):
    '''
    The + operator of Lisp. Sums a bunch of numbers.

    (+ <x> <y> <z> ...)
    '''
    return __builtins__.sum(args)


@meta()
def sub(*args, **kwargs):
    '''
    The - operator of Listp. Substracts a bunch of numbers.

    (- <x> <y> <z> ...)
    '''
    if len(args) == 0:
        return 0
    
    value = args[0]

    if len(args) == 1:
        return -value

    for i in args[1:]:
        value -= i
    
    return value


@meta()
def mul(x, y, *args, **kwargs):
    '''
    The * operator of Lisp. Multiplies a bunch of numbers.

    (* <x> <y> <z> ...)
    '''
    value = x * y

    for i in args:
        value *= i
    
    return value


@meta()
def div(x, y, *args, **kwargs):
    '''
    The / operator of Lisp. Divides a bunch of numbers.

    (/ <x> <y> <z> ...)
    '''
    value = x / y

    for i in args:
        value /= i
    
    return value


@meta()
def greater_than(x, y, **kwargs):
    '''
    The > operator of Lisp. Says if something is greater than something else.

    (> <x> <y>)
    '''
    return x > y


@meta()
def greater_than_eq(x, y, **kwargs):
    '''
    The >= operator of Lisp. Says if something is greater than or equal something else.

    (>= <x> <y>)
    '''
    return x >= y


@meta()
def eq(x, y, **kwargs):
    '''
    The = Operator of Lisp. Says if something is equal to something else.

    (= <x> <y>)
    '''
    return x == y


@meta()
def lesser_than(x, y, **kwargs):
    '''
    The < operator of Lisp. Says if something is less than something else.

    (< <x> <y>)
    '''
    return x < y


@meta()
def lesser_than_eq(x, y, **kwargs):
    '''
    The <= operator of Lisp. Says if something is less than or equal something else.

    (<= <x> <y>)
    '''
    return x <= y


@meta([
    (1, { 'evaluate': False }),
    (2, { 'evaluate': False }),
])
def _if(predicate, consequent, alternative=None, scope={}):
    '''
    The if operator of Lisp. Evaluates a predicate, and if it's true evaluates
    a consequent or an alternative otherwise.

    (if <predicate> <consequent> <alternative>)
    (if <predicate> <consequent>)
    '''
    if predicate:
        return runtime.eval(consequent, scope)
    elif alternative is not None:
        return runtime.eval(alternative, scope)
    else:
        return None


@meta([
    (0, { 'evaluate': False }),
    (1, { 'evaluate': False })
])
def _lambda(args, body, scope):
    '''
    The lambda operator of Lisp. Defines a lambda function with args and a body.

    (lambda (<x> <y> ...) <body>)
    '''
    arg_names = list(map(lambda x: x.value, args.value))
    
    @meta()
    def f(*args, **kwargs):
        if len(arg_names) != len(args):
            raise RuntimeError('expected %d args, received %d' % (len(arg_names), len(args)))
        
        local = dict(zip(arg_names, args))
        return runtime.eval(body, scope=kwargs['scope'] + local)
    
    return f


# The initial scope, with the main operators.
# To define new operators, define a function with
# the @meta decorator.
scope = {
    'define': define,
    '+': sum,
    '-': sub,
    '*': mul,
    '/': div,
    '>': greater_than,
    '>=': greater_than_eq,
    '=': eq,
    '<': lesser_than,
    '<=': lesser_than_eq,
    'if': _if,
    'lambda': _lambda
}

runtime = VM(scope)


def print_help():
    print('''
    lisper.py [options] program.lisp

    -t - print tokens
    -a - print AST
    -i - print each evaluation
    -n - does not evaluate
    -h - print help
    ''')


def main(argv):
    opts, args = getopt.getopt(argv, 'tahin')
    print_tokens = False
    print_ast = False
    print_iterations = False
    evaluate = True

    for opt, arg in opts:
        if opt == '-t':
            print_tokens = True
        elif opt == '-a':
            print_ast = True
        elif opt == '-h':
            print_help()
            sys.exit(0)
        elif opt == '-i':
            print_iterations = True
        elif opt == '-n':
            evaluate = False
        
    if len(args) != 1:
        raise SystemError('You need to pass one program to run')

    tokenizer = Tokenizer(open(args[0]).read())
    tokens = tokenizer.tokenize()

    if print_tokens:
        print(json.dumps(tokens.serialize(), indent=4))

    parser = Parser(tokens)
    ast = parser.parse()

    if print_ast:
        print(json.dumps(ast.serialize(), indent=4))

    runtime.print_i = print_iterations
    if evaluate:
        runtime.eval(ast)
        print(runtime.current_runtime_value)



if __name__ == '__main__':
    main(sys.argv[1:])
