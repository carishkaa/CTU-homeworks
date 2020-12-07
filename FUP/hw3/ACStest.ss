#lang r5rs
 
;;; ------------ STATE COMPONENTS ------------
 
;;; get maze
(define (maze state) (car state))
 
;;; get coordinates list
(define (coordinates state) (cadr state))
 
;;; get x coordinate
(define (coordinate-x state) (car (coordinates state)))
 
;;; get y coordinate
(define (coordinate-y state) (cadr (coordinates state)))
 
;;; get orientation
(define (orientation state) (caddr state))
 
;;; get action sequence
(define (actions state)
  (if (null? (cdddr state)) '() (car (cdddr state)))
)
 
 
;;; ------------ IF-CONDITIONS ------------
 
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
 
;;; ------------ HELPER FUNCTIONS ------------
 
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
 
; change element of list
(define (change-element lst pos val)
  (cond ((= pos 0) (cons val (cdr lst)))
        (#t (cons (car lst) (change-element (cdr lst) (- pos 1) val)))
    )
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
 
 
;;; ------------ ACTIONS ------------
 
;;; Step
(define (step state expr program limit lenlimit)
  (define x (coordinate-x state))
  (define y (coordinate-y state))
  (if (wall? state) (append (list 'exit) state) ;cannot be done -> program emds
      (cond
        ((north? state) (main-func (list (maze state) (list x (- y 1)) (orientation state) (append (actions state) '(step))) (expr-upd expr) program limit lenlimit))
        ((south? state) (main-func (list (maze state) (list x (+ y 1)) (orientation state) (append (actions state) '(step))) (expr-upd expr) program limit lenlimit))
        ((east? state) (main-func (list (maze state) (list (+ x 1) y) (orientation state) (append (actions state) '(step))) (expr-upd expr) program limit lenlimit))
        ((west? state) (main-func (list (maze state) (list (- x 1) y) (orientation state) (append (actions state) '(step))) (expr-upd expr) program limit lenlimit)))))
 
 
;;; Turn left
(define (turn-left state expr program limit lenlimit)
  (main-func
   (list (maze state) (coordinates state) (left-orientation state) (append (actions state) '(turn-left)))
   (expr-upd expr) program limit lenlimit)
)
 
;;; Put mark
(define (put-mark state expr program limit lenlimit)
  (main-func
   (list (change-field inc (coordinate-x state) (coordinate-y state) (maze state)) (coordinates state) (orientation state) (append (actions state) '(put-mark)))
   (expr-upd expr) program limit lenlimit)
)
 
;;; Get mark
(define (get-mark state expr program limit lenlimit)
  (if
   (eqv? (cur-field state) 0) (append (list 'exit) state) ; no marks -> the action can't be done -> program ends
   (main-func (list (change-field dec (coordinate-x state) (coordinate-y state) (maze state)) (coordinates state) (orientation state) (append (actions state) '(get-mark)))
           (expr-upd expr) program limit lenlimit))
)
 
 
;;; ------------ MAIN FUNCTION ------------
 
(define (main-func state expr program limit lenlimit)
  (cond
    ; limit is exceeded -> exit
    ((eqv? (car state) 'exit) state)
 
    ; too much actions
    ((> (length (actions state)) lenlimit) (append (list 'exit) state))
     
    ; empty list -> nop
    ((and (list? expr) (null? expr)) state)
     
    ; commands
    ((eqv? expr 'step) (step state expr program limit lenlimit))
    ((eqv? expr 'turn-left) (turn-left state expr program limit lenlimit))
    ((eqv? expr 'put-mark) (put-mark state expr program limit lenlimit))
    ((eqv? expr 'get-mark) (get-mark state expr program limit lenlimit))
 
    ; procedures
    ((not (list? expr))
     (if (= limit 0) (append (list 'exit) state) ;limit is exceeded 
       (main-func state (find-procedure expr program) program (- limit 1) lenlimit)) ; call procedure 
     )
     
    ; if-condition: (if <condition> <positive-branch> <negative-branch>)
    ((eqv? (car expr) 'if)
     (let ((condition (cadr expr))
           (positive-branch (caddr expr))
           (negative-branch (cadr (cddr expr))))
       (cond
       ((eqv? condition 'wall?)  (if (wall? state)  (main-func state positive-branch program limit lenlimit) (main-func state negative-branch program limit lenlimit)))
       ((eqv? condition 'north?) (if (north? state) (main-func state positive-branch program limit lenlimit) (main-func state negative-branch program limit lenlimit)))
       ((eqv? condition 'mark?)  (if (mark? state)  (main-func state positive-branch program limit lenlimit) (main-func state negative-branch program limit lenlimit)))
       ))
     )
 
    ; some sequence
    (else (main-func (main-func state (car expr) program limit lenlimit) (cdr expr) program limit lenlimit))
  )
)
 
;;; ------------ SIMULATE ------------
(define (simulate state expr program limit lenlimit)
  (define result (main-func state expr program limit lenlimit))
  (if (eqv? (car result) 'exit)
      (list (actions (cdr result)) (list (maze (cdr result)) (coordinates (cdr result)) (orientation (cdr result))))
      (list (actions result) (list (maze result) (coordinates result) (orientation result)))
   )
)
 
;;;  ------------ HW02 ------------
 
;;; ------------ COMPONENTS ------------
 
;;; Manhattan distance
(define (manhattan-dist maze1 maze2)
  (define (marks-diff x y) (map (lambda (x y) (map (lambda (x y) (if (eqv? x 'w) 0 (abs (- x y)))) x y)) x y))
  (define (sum elemList) (if (null? elemList) 0 (+ (apply + (car elemList)) (sum (cdr elemList)))))
  (sum (marks-diff maze1 maze2))
)
 
;;; Configuration distance
(define (config-dist state1 state2)
  (+ (abs (- (coordinate-x state1) (coordinate-x state2))) (abs (- (coordinate-y state1) (coordinate-y state2))) (if (eq? (orientation state1) (orientation state2)) 0 1))
)
 
;;; The length of the program
(define (len prg)
  (cond
    ((pair? prg) (+ (len (car prg)) (len (cdr prg))))
    ((null? prg) 0)
    ((or (eqv? prg 'procedure) (eqv? prg 'if)) 0)
    (else 1))
)
 
  
;;; ------------ EVALUATE ------------
 
;;; evaluate
(define (evaluate prgs pairs threshold stack_size)
  ;(newline) (display 'evaluate) (newline) (display (in-eval prgs pairs threshold stack_size)) (newline) (display 'sorted:) (newline)
  ;(display (bubble-sort (in-eval prgs pairs threshold stack_size)))
  (bubble-sort (in-eval prgs pairs threshold stack_size))
)
 
; go through all the programs
(define (in-eval prgs pairs threshold stack-size)
  (cond
    ((null? prgs) '())
    ((> (len (car prgs)) (list-ref threshold 2)) (in-eval (cdr prgs) pairs threshold stack-size))
    (else (append (calculate-prgvalue (car prgs) pairs threshold stack-size '(0 0 0 0)) (in-eval (cdr prgs) pairs threshold stack-size)))
  )
)
 
; run program for all maze pairs, returns (<value> <program>)
(define (calculate-prgvalue prg pairs threshold stack-size prgvalue)
  (cond
    ((out-of-limit prgvalue threshold) '())
    ((null? pairs) (list (cons (change-element prgvalue 2 (len prg)) (list prg))))
    (else (calculate-prgvalue prg (cdr pairs) threshold stack-size (map + prgvalue (start-simulation prg (car pairs) threshold stack-size))))
    )
  )
 
; starts a simulation with a specific maze and program
(define (start-simulation prg pair threshold stack-size)
  (let* ((result (simulate (car pair) 'start prg stack-size (list-ref threshold 3)))
         (final-state (cadr result))
         (desired-state (cadr pair)))
    (list
     (manhattan-dist (maze final-state) (maze desired-state))
     (config-dist final-state desired-state)
     0
     (length (car result))
     ))
)
 
(define (out-of-limit prgvalue threshold)
  (cond
    ((not (pair? prgvalue)) #f)
    ((> (car prgvalue) (car threshold)) #t)
    (else (out-of-limit (cdr prgvalue) (cdr threshold)))
  )
)
    
;;; ------------ SORT ------------
(define (bubble-sort lst) (bubble-in lst (length lst)))
 
(define (bubble-in lst n)
  (if (= n 1) lst (bubble-in (bubble-swap lst) (- n 1))))
 
(define (bubble-swap ls)
  (if (null? (cdr ls))
      ls
      (if (compare (cadr ls) (car ls))
          (cons (cadr ls) (bubble-swap (cons (car ls) (cddr ls))))
          (cons (car ls) (bubble-swap (cdr ls))))))
 
(define compare (lambda (x y)
       (if (not (eq? (car (car x)) (car (car y)))) (< (car (car x)) (car (car y)))
           (if (not (eq? (car (cdr (car x))) (car (cdr (car y))))) (< (car (cdr (car x))) (car (cdr (car y))))
               (if (not (eq? (car (cddr (car x))) (car (cddr (car y))))) (< (car (cddr (car x))) (car (cddr (car y))))
                   (< (car (cddr (cdr (car x)))) (car (cddr (cdr (car y))))))))))
 
 
;;; ------------ HW03 ------------
 
(define (congruential-rng seed)
  (let ((a 16807 #|(expt 7 5)|#)
        (m 2147483647 #|(- (expt 2 31) 1)|#))
    (let ((m-1 (- m 1)))
      (let ((seed (+ (remainder seed m-1) 1)))
        (lambda (b)
          (let ((n (remainder (* a seed) m)))
            (set! seed n)
            (quotient (* (- n 1) b) m-1)))))))
(define random (congruential-rng 12345))
 
(define population-size 30)
(define initial-population '(
    ((procedure start ()))
    ((procedure start (start)))
    ((procedure start (step)))
    ((procedure start (step start)))
    ((procedure start (put-mark)))
    ((procedure start (turn-left turn-left turn-left)))
    ((procedure start ((if wall? () step))))
    ((procedure start ((if wall? (turn-left (if wall? turn-left step)) step))))
    ((procedure start ((if mark? get-mark put-mark) step)))
    ((procedure start ((if north? (start) ()) turn-left)))
    ((procedure start ((if mark? (put-mark step) (start)))))
    ((procedure start (step (if north? turn-left step))))
    ((procedure start (step step step put-mark)))
))
 
; -------- RANDOM LISTS --------
 
(define if-conds '(mark? north? wall?))
 
(define commands '(start put-mark get-mark turn-left step))
 
(define mutations (list 
   (lambda (prg) (add-cmd prg (random (length prg))))
   (lambda (prg) (add-cmd prg (random (length prg))))
   (lambda (prg) (if (> (length prg) 1) (remove-cmd prg (random (length prg))) '()))
   (lambda (prg) (if (> (length prg) 1) (remove-cmd prg (random (length prg))) '()))
   (lambda (prg) (cons (add-cmd prg (random (length prg))) (add-cmd prg (random (length prg)))))
))

(define operations (list 
   (lambda (prgs) (mutation prgs))
   ; + crossover + copy
))
 
; ------ HELPERS -------
 
(define (get-random-from lst)
  (n-th (random (length lst)) lst)
)
 
(define n-th (lambda (i lst) (list-ref lst i)))
 
(define (n-first lst n)
    (if (= n 0) '() (cons (car lst) (n-first (cdr lst) (- n 1))))
)
 
(define (get-random-prg prgs)
  (caddar (get-random-from prgs))
)
 
; ----------- PROGRAM MUTATION -----------

(define (mutation prgs)
  (let ((prg (get-random-prg prgs))
        (mutate (get-random-from mutations)))
    (list (append '(procedure start) (list (mutate prg)))))
)

; add random command or if
(define (add-cmd prg position)
  (cond
    ((null? prg) '())
    ((= position 0) (cons (create-cmd) prg))
    (else (cons (car prg) (add-cmd (cdr prg) (- position 1))))
  )
)
 
(define (create-cmd)
  (if (zero? (random 2))
  (list 'if (get-random-from if-conds) (get-if-expr) (get-if-expr))
  (get-random-from commands)
  )
)
 
(define (get-if-expr)
  (if (= 0 (random 10)) '() (list (get-random-from commands))) 
)
 
; remove random command or if
(define (remove-cmd prg position)
  (if (= position 0) (cdr prg) (cons (car prg) (remove-cmd (cdr prg) (- position 1))))
)
 
;;; ------------ EVOLVE ------------
 
(define (evolve pairs threshold stack-size)
  (in-evolve initial-population pairs threshold stack-size)
)
 
(define (in-evolve prgs pairs threshold stack-size)
  (let* ((evaluated (evaluate prgs pairs threshold stack-size))
         (best-results (n-first evaluated 5))
         (best-prgs (map cadr best-results)))
    (display (car best-results))
    (newline)
    (in-evolve (new-population best-prgs prgs) pairs threshold stack-size)
  )
)
 
(define (new-population best-prgs all-prgs)
  (let* ((generated-prgs (generate-n-prgs (- population-size (length best-prgs)) all-prgs)))
    (append best-prgs generated-prgs)
))
 
(define (generate-n-prgs n prgs)
  (let ((create-new (get-random-from operations)))
    (if (= n 0) (list (create-new prgs)) (cons (create-new prgs) (generate-n-prgs (- n 1) prgs)))
  )
)

;------------ Tests ------------
; PUBLIC 1
  (define pairs1
    '(
      (
       (((w w w) 
         (w 0 w) 
         (w w w))  
        (1 1) west)
       
       (((w w w) 
         (w 1 w) 
         (w w w))  
        (1 1) west)
       )
      (
       (((w w w w) 
         (w 0 w w) 
         (w 0 w w) 
         (w 0 w w) 
         (w 0 0 w) 
         (w w w w)) 
        (1 4) north)
       
       (((w w w w) 
         (w 1 w w) 
         (w 0 w w) 
         (w 0 w w) 
         (w 0 0 w) 
         (w w w w)) 
        (1 1) north)
       )
      (
       (((w w w w w w) 
         (w 0 w w w w) 
         (w 0 w 3 0 w) 
         (w 1 3 0 w w) 
         (w w w w w w)) 
        (3 2) east)
       
       (((w w w w w w) 
         (w 0 w w w w) 
         (w 0 w 3 1 w) 
         (w 1 3 0 w w) 
         (w w w w w w))
        (4 2) east)
       )
      (
       (((w w w w w w) 
         (w 2 3 0 w w) 
         (w w w w w w)) 
        (3 1) west)
       
       (((w w w w w w) 
         (w 3 3 0 w w) 
         (w w w w w w))
        (1 1) west)
       )
      )
    )


(define pairs2
    '(
      (
       (((w w w) 
         (w 1 w) 
         (w w w))  
        (1 1) west)
       
       (((w w w) 
         (w 1 w) 
         (w w w))  
        (1 1) north)
       )
      (
       (((w w w w w w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 w w) 
         (w w w w w w)) 
        (2 4) north)
       
       (((w w w w w w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 w w) 
         (w w w w w w)) 
        (2 4) north)
       )
      (
       (((w w w w w w) 
         (w 0 0 0 0 w) 
         (w 0 0 3 1 w) 
         (w 1 3 0 0 w) 
         (w w w w w w)) 
        (3 2) east)
       
       (((w w w w w w) 
         (w 0 0 0 0 w) 
         (w 0 0 3 1 w) 
         (w 1 3 0 0 w) 
         (w w w w w w))
        (3 2) north)
       )
      (
       (((w w w w w w) 
         (w 3 3 0 0 w) 
         (w w w w w w)) 
        (3 1) south)
       
       (((w w w w w w) 
         (w 3 3 0 0 w) 
         (w w w w w w))
        (3 1) north)
       )
      )
    )

(define pairs3
    '(
      (
       (((w w w) 
         (w 1 w) 
         (w w w))  
        (1 1) west)
       
       (((w w w) 
         (w 1 w) 
         (w w w))  
        (1 1) north)
       )
      (
       (((w w w w w w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 w w) 
         (w w w w w w)) 
        (2 4) north)
       
       (((w w w w w w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 0 w) 
         (w 0 0 0 w w) 
         (w w w w w w)) 
        (2 4) east)
       )
      (
       (((w w w w w w) 
         (w 0 0 0 0 w) 
         (w 0 0 3 1 w) 
         (w 1 3 0 0 w) 
         (w w w w w w)) 
        (3 2) east)
       
       (((w w w w w w) 
         (w 0 0 0 0 w) 
         (w 0 0 3 1 w) 
         (w 1 3 0 0 w) 
         (w w w w w w))
        (3 2) south)
       )
      (
       (((w w w w w w) 
         (w 3 3 0 0 w) 
         (w w w w w w)) 
        (3 1) south)
       
       (((w w w w w w) 
         (w 3 3 0 0 w) 
         (w w w w w w))
        (3 1) west)
       )
      )
    )
(newline) 
(evolve pairs1 '(1000 1000 20 20) 8)
;(evolve pairs2 '(1000 1000 20 20) 5)
;(evolve pairs3 '(1000 1000 20 20) 1)
