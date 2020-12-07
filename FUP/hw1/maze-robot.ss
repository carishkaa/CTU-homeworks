;#lang r5rs

;;; _______STATE COMPONENTS______

;;; get maze
(define (maze state)
  (car state))

;;; get coordinates list
(define (coordinates state)
  (cadr state))

;;; get x coordinate
(define (coordinate-x state)
  (car (coordinates state)))

;;; get y coordinate
(define (coordinate-y state)
  (cadr (coordinates state)))

;;; get orientation
(define (orientation state)
  (caddr state))

;;; get action sequence
(define (actions state)
  (cond ((null? (cdddr state)) '())
        (else (car (cdddr state)))
))


;;;________ IF-CONDITIONS ________

;;; orientation
(define (north? state)
  (eqv? (orientation state) 'north))

(define (east? state)
  (eqv? (orientation state) 'east))

(define (west? state)
  (eqv? (orientation state) 'west))

(define (south? state)
  (eqv? (orientation state) 'south))

;;; does the field have marks?
(define (mark? state)
  (not (eqv? (cur-field state) 0)))

;;; is there a wall in front of the robot?
(define (wall? state)
  (define x (coordinate-x state))
  (define y (coordinate-y state))
  (cond
    ((north? state) (eqv? (field x (- y 1) state) 'w))
    ((south? state) (eqv? (field x (+ y 1) state) 'w))
    ((east? state)  (eqv? (field (+ x 1) y state) 'w))
    ((west? state)  (eqv? (field (- x 1) y state) 'w))
    )
)

;;; ________ HELPER FUNCTIONS ________

;;; get field type of [x,y] element in maze list
(define (field x y state)
  (list-ref (list-ref (maze state) y) x))

(define (cur-field state)
  (field (coordinate-x state) (coordinate-y state) state))

;;; change value at field
(define (change-field fn x y maze)
  (define (apply-at fn list pos)
    (cond ((= pos 0) (cons (fn (car list)) (cdr list)))
        (#t (cons (car list) (apply-at fn (cdr list) (- pos 1))))
    ))
  (apply-at (lambda (line) (apply-at fn line x)) maze y)
)

;;; find the procedure in program
(define (find-procedure name program)
  (cond
    ((null? program) '()) ; no program name 
    ((and (eqv? (caar program) 'procedure) (eqv? (cadr (car program)) name)) (cddar program)) ; found
    (else (find-procedure name (cdr program))) ; continue by recursive
))

;;; next orientation (for turn-left)
(define (left-orientation state)
  (cond
    ((north? state) 'west)
    ((south? state) 'east)
    ((east? state) 'north)
    ((west? state) 'south)
))

;;; increment the number
(define (inc n)
  (+ n 1))

;;; decrement the number
(define (dec n)
  (- n 1))

;;; return the rest of expr list
(define (expr-upd expr)
  (if (and (not (null? expr)) (not (pair? expr))) '() (cdr expr))
)


;;; __________ ACTIONS __________

;;; Step
(define (step state expr program limit)
  (define x (coordinate-x state))
  (define y (coordinate-y state))
  (if (wall? state) (append (list 'exit) state) ;cannot be done -> program emds
      (cond
        ((north? state) (main-func (list (maze state) (list x (- y 1)) (orientation state) (append (actions state) '(step))) (expr-upd expr) program limit))
        ((south? state) (main-func (list (maze state) (list x (+ y 1)) (orientation state) (append (actions state) '(step))) (expr-upd expr) program limit))
        ((east? state) (main-func (list (maze state) (list (+ x 1) y) (orientation state) (append (actions state) '(step))) (expr-upd expr) program limit))
        ((west? state) (main-func (list (maze state) (list (- x 1) y) (orientation state) (append (actions state) '(step))) (expr-upd expr) program limit)))))


;;; Turn left
(define (turn-left state expr program limit)
  (main-func
   (list (maze state) (coordinates state) (left-orientation state) (append (actions state) '(turn-left)))
   (expr-upd expr) program limit)
)

;;; Put mark
(define (put-mark state expr program limit)
  (main-func
   (list (change-field inc (coordinate-x state) (coordinate-y state) (maze state)) (coordinates state) (orientation state) (append (actions state) '(put-mark)))
   (expr-upd expr) program limit)
)

;;; Get mark
(define (get-mark state expr program limit)
  (if
   (eqv? (cur-field state) 0) (append (list 'exit) state) ; no marks -> the action can't be done -> program ends
   (main-func (list (change-field dec (coordinate-x state) (coordinate-y state) (maze state)) (coordinates state) (orientation state) (append (actions state) '(get-mark)))
           (expr-upd expr) program limit))
)


;;; ________ MAIN FUNCTION ________
(define (main-func state expr program limit)
  (cond
    
    ; limit is exceeded -> exit
    ((eqv? (car state) 'exit) state)
    
    ; empty list -> nop
    ((and (list? expr) (null? expr)) state)
    
    ; commands
    ((eqv? expr 'step) (step state expr program limit))
    ((eqv? expr 'turn-left) (turn-left state expr program limit))
    ((eqv? expr 'put-mark) (put-mark state expr program limit))
    ((eqv? expr 'get-mark) (get-mark state expr program limit))

    ; procedures
    ((not (list? expr))
     (if (= limit 0) (append (list 'exit) state) ;limit is exceeded
       (main-func state (find-procedure expr program) program (- limit 1))) ; call procedure 
     )
    
    ; if-condition: (if <condition> <positive-branch> <negative-branch>)
    ((eqv? (car expr) 'if)
     (let ((condition (cadr expr))
           (positive-branch (caddr expr))
           (negative-branch (cadr (cddr expr))))
       (cond
       ((eqv? condition 'wall?)  (if (wall? state)  (main-func state positive-branch program limit) (main-func state negative-branch program limit)))
       ((eqv? condition 'north?) (if (north? state) (main-func state positive-branch program limit) (main-func state negative-branch program limit)))
       ((eqv? condition 'mark?)  (if (mark? state)  (main-func state positive-branch program limit) (main-func state negative-branch program limit)))
       ))
     )

    ; some sequence
    (else (main-func (main-func state (car expr) program limit) (cdr expr) program limit))
  )
)

;;; SIMULATE
(define (simulate state expr program limit)
  (define ret (main-func state expr program limit))
  (if (eqv? (car ret) 'exit)
      (list (actions (cdr ret)) (list (maze (cdr ret)) (coordinates (cdr ret)) (orientation (cdr ret))))
      (list (actions ret) (list (maze ret) (coordinates ret) (orientation ret)))
   )
)

;;; ________ TESTS ________
