#!/usr/bin/env python

import sys
import json
import getopt

# Defines a set of valid non-alphanumeric characters
# for symbols. It's a constant in order to be a O(1)
# query evertime the lexer looks for it.
VALID_SYMBOL_CHARS = set('!-?+/*<>=._')


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

        value = self.current().value

        if value.find('.') >= 0:
            return float(value)

        return int(value)

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
                expression.append(Symbol(self.current().value))
                self.consume(1)
            elif self.current().type == 'NUMBER':
                expression.append(self.parse_number())
                self.consume(1)
            elif self.current().type == 'RPAREN':
                self.consume(1)
                return expression
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


class Symbol:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return '\'' + str(self)


class RT:
    def __init__(self):
        self.vars = {
            'define': self.define,
            'lambda': self.lambda_,
        }

    def eval(self, node, scope={}):
        if isinstance(node, Program):
            for n in node.nodes:
                self.eval(n)

            return

        if isinstance(node, list):
            return self.eval_call(node, scope)

        if isinstance(node, Symbol):
            return self.var(str(node), scope)

        return node

    def eval_args(self, args, scope):
        return list(map(lambda x: self.eval(x, scope), args))

    def eval_call(self, node, scope):
        call_name = node[0]

        if isinstance(call_name, Symbol):
            if str(call_name).startswith('.'):
                value = self.eval(node[1], scope)
                fn = getattr(value, str(call_name)[1:])

                return fn(*self.eval_args(node[2:], scope))
            elif str(call_name) == 'lambda':
                return self.lambda_(node, scope)
            elif str(call_name) == 'define':
                return self.define(node, scope)
            elif str(call_name) == 'if':
                return self.if_(node, scope)
            else:
                fn = self.var(call_name, scope)
        else:
            fn = self.eval(call_name, scope)

        return fn(*self.eval_args(node[1:], scope))

    def var(self, name, scope):
        if str(name).startswith('py/'):
            return getattr(__builtins__, str(name)[3:])

        if scope.get(str(name), None) is not None:
            return scope[str(name)]

        return self.vars[str(name)]

    def define(self, node, scope):
        name = str(node[1])
        value = self.eval(node[2], scope)

        self.vars[name] = value

        return value

    def lambda_(self, node, scope):
        names = list(map(str, node[1]))
        forms = node[2:]

        def fn(*args):
            local_scope = dict(zip(names, args), **scope)

            value = None

            for form in forms:
                value = self.eval(form, local_scope)

            return value

        return fn

    def if_(self, node, scope):
        predicate = node[1]
        affirmative = node[2]
        negative = node[3]

        if self.eval(predicate, scope):
            return self.eval(affirmative, scope)
        else:
            return self.eval(negative, scope)


def print_help():
    print('''
    lisper.py [options] program.lisp

    -t - print tokens
    -a - print AST
    -i - print each evaluation
    -n - does not evaluate
    -h - print help
    ''')


def get_core_module():
    tokens = Tokenizer(open('core.lisp').read()).tokenize()

    return Parser(tokens).parse()


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

    runtime = RT()

    if evaluate:
        runtime.eval(get_core_module())
        runtime.eval(ast)


if __name__ == '__main__':
    main(sys.argv[1:])
