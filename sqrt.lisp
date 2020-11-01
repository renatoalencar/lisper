; Solves a square root applying the Heron of Alexandria's method.

; define the square of x
(define square
    (lambda (x) (* x x)))

; define the average between x and y
(define average
    (lambda (x y)
            (/ (+ x y) 2)))

; improve the guess
(define improve
    (lambda (guess x)
            (average guess (/ x guess))))

; check if the guess is good enough
(define good-enough?
    (lambda (guess x)
            (< (py/abs (- (square guess) x))
            0.00000001)))

; try a guess
(define try
    (lambda (guess x)
            (if (good-enough? guess x)
                guess
                (try (improve guess x) x))))

; take the square root of a number making the first
; guess as one
(define sqrt (lambda (x) (try 1 x)))

; test
(py/print (sqrt 2.0))
(py/print (py/isinstance 2.0 py/float))
