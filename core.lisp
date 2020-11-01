
(define <
    (lambda (x y)
      (.__lt__ x y)))

(define -
    (lambda (x y)
      (.__sub__ (.__float__ x) y)))

(define *
    (lambda (x y)
      (.__mul__ x y)))

(define /
    (lambda (x y)
      (.__truediv__ x y)))

(define +
    (lambda (x y)
      (.__add__ (.__float__ x) y)))
