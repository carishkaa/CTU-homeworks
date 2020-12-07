#lang scheme

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


;;; __________ ACTIONS __________

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


;;; ________ MAIN FUNCTION ________

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

;;; ____________ SIMULATE ____________
(define (simulate state expr program limit lenlimit)
  (define result (main-func state expr program limit lenlimit))
  (if (eqv? (car result) 'exit)
      (list (actions (cdr result)) (list (maze (cdr result)) (coordinates (cdr result)) (orientation (cdr result))))
      (list (actions result) (list (maze result) (coordinates result) (orientation result)))
   )
)


;;; ____________ COMPONENTS ____________

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

 
;;; ____________ EVALUATE ____________


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

; run program for all maze pairs, returns (<value> <program>)
(define (calculate-prgvalue prg pairs threshold stack-size prgvalue)
  (cond
    ((out-of-limit prgvalue threshold) '())
    ((null? pairs) (list (cons (change-element prgvalue 2 (len prg)) (list prg))))
    (else (calculate-prgvalue prg (cdr pairs) threshold stack-size (map + prgvalue (start-simulation prg (car pairs) threshold stack-size))))
    )
  )

; go through all the programs
(define (my-eval prgs pairs threshold stack-size)
  (cond
    ((null? prgs) '())
    ((> (len (car prgs)) (list-ref threshold 2)) (my-eval (cdr prgs) pairs threshold stack-size))
    (else (append (calculate-prgvalue (car prgs) pairs threshold stack-size '(0 0 0 0)) (my-eval (cdr prgs) pairs threshold stack-size)))
  )
)

;;; evaluate
(define (evaluate prgs pairs threshold stack_size)
  (bubble-sort (my-eval prgs pairs threshold stack_size))
)

;;; ____________ SORT ____________
(define (bubble-sort lst)
  (letrec
    (
     (fix (lambda (f i)
       (if (equal? i (f i))
           i
           (fix f (f i)))))
 
     (sort-step (lambda (lst)
        (if (or (null? lst) (null? (cdr lst)))
            lst
            (if (compare (car lst) (cadr lst))
                (cons (car  lst) (sort-step (cdr lst)))
                (cons (cadr lst) (sort-step (cons (car lst) (cddr lst))))
                ))))
     
     (compare (lambda (x y)
       (if (not (eq? (caar x) (caar y))) (< (caar x) (caar y)) 
           (if (not (eq? (car (cdar x)) (cadar y))) (< (cadar x) (cadar y))
               (if (not (eq? (car (cddar x)) (car (cddr (car y))))) (< (car (cddar x)) (car (cddar y)))
                   (< (car (cdddr (car x))) (car (cdddr (car y)))))))))
     )
  (fix sort-step lst)))


;__________Tests____________
; Assert function

(define (assert-equal fun res exp)

  (cond ((equal? res exp) (display (list "TEST PASSED, called:" fun "and received" res)))

        (else (display (list "TEST FAILED, called:" fun)) (newline) (display "Recieved:") (newline) (display res) (newline) (display "Expected:") (newline) (display exp))

      )

  (newline)

  )
; Tests of evaluate
; PUBLIC 1
(newline)
(display "PUB01")

(define prgs
'(
   (
      (procedure start
         (turn-right (if wall? (turn-left 
             (if wall? (turn-left (if wall? turn-left step)) step)) step)
                 put-mark start )
      )
      (procedure turn-right (turn-left turn-left turn-left))
  )
  (
      (procedure start  (put-mark (if wall? turn-left step) start))
  )
  (
      (procedure start (step step step put-mark))
  )
)
)


(define pairs
'(
  (
   (((w w w w w w) 
     (w 0 w 0 w w) 
     (w 1 w 0 0 w) 
     (w 1 0 0 w w) 
     (w w w w w w)) 
     (1 3) south)

   (((w w w w w w) 
     (w 0 w 0 w w) 
     (w 0 w 0 0 w) 
     (w 0 0 0 w w) 
     (w w w w w w)) 
     (1 1) north)
   )
   (
   (((w w w w w w) 
     (w 0 w 0 w w) 
     (w 0 w 2 0 w) 
     (w 1 3 0 w w) 
     (w w w w w w)) 
     (3 3) north)

   (((w w w w w w) 
     (w 0 w 0 w w) 
     (w 0 w 0 0 w) 
     (w 0 0 0 w w) 
     (w w w w w w)) 
     (1 1) north)
  ))
 )

(assert-equal "(evaluate prgs pairs '(20 20 20 20) 5)" (evaluate prgs pairs '(20 20 20 20) 5) '(
((8 5 5 2) ((procedure start (step step step put-mark))))
((18 7 6 20) ((procedure start (put-mark (if wall? turn-left step) start))))
))

; PUBLIC 2
(newline)
(display "PUB02")

(define s0
'(
(w w w)
(w 0 w)
(w w w)
)
)

(define t0
'(
(w w w)
(w 1 w)
(w w w)
)
)

(define s1
'(
(w w w)
(w 0 w)
(w 0 w)
(w w w)
)
)

(define t1
'(
(w w w)
(w 0 w)
(w 1 w)
(w w w)
)
)

(define s2
'(
(w w w)
(w 0 w)
(w 0 w)
(w 0 w)
(w 0 w)
(w 0 w)
(w 0 w)
(w 0 w)
(w w w)
)
)

(define t2
'(
(w w w)
(w 0 w)
(w 0 w)
(w 0 w)
(w 0 w)
(w 0 w)
(w 0 w)
(w 1 w)
(w w w)
)
)

(define p1
  '(
    (procedure start
      put-mark
    )
   )
)

(define p2
  '(
    (procedure start
      (if wall? put-mark step)
    )
   )
)

(define p3
  '(
    (procedure start
      (if wall? put-mark (step start))
    )
   )
)

(define p4
  '(
    (procedure start
      (if wall? put-mark (step start turn-left turn-left step turn-left turn-left))
    )
   )
)

(newline)
(assert-equal "(evaluate `(,p1 ,p2 ,p3 ,p4) `(((,s0 (1 1) south) (,t0 (1 1) south)) ((,s1 (1 1) south) (,t1 (1 1) south)) ((,s2 (1 1) south) (,t2 (1 1) south))) `(50 50 50 50) 20)" (evaluate `(,p1 ,p2 ,p3 ,p4) `(((,s0 (1 1) south) (,t0 (1 1) south)) ((,s1 (1 1) south) (,t1 (1 1) south)) ((,s2 (1 1) south) (,t2 (1 1) south))) `(50 50 50 50) 20) '(
((0 0 10 45) ((procedure start (if wall? put-mark (step start turn-left turn-left step turn-left turn-left)))))
((0 7 5 10) ((procedure start (if wall? put-mark (step start)))))
((2 2 4 3) ((procedure start (if wall? put-mark step))))
((4 0 2 3) ((procedure start put-mark)))
))

;PUBLIC 3
(newline)
(display "PUB03")

(define s0-3
'(
(w w w)
(w 3 w)
(w w w)
)
)

(define t0-3
'(
(w w w)
(w 3 w)
(w w w)
)
)

(define s1-3
'(
(w w w)
(w 4 w)
(w 4 w)
(w w w)
)
)

(define t1-3
'(
(w w w)
(w 4 w)
(w 4 w)
(w w w)
)
)

(define s2-3
'(
(w w w w w w w w)
(w 0 0 0 0 0 0 w)
(w 0 0 0 0 0 0 w)
(w 0 w w 0 w 0 w)
(w 0 w 0 0 w 0 w)
(w 0 w 0 0 w 0 w)
(w 0 w w w w 0 w)
(w 0 0 0 0 0 0 w)
(w 0 0 0 0 0 0 w)
(w w w w w w w w)
)
)

(define t2-3
'(
(w w w w w w w w)
(w 0 0 0 0 0 0 w)
(w 0 0 0 0 0 0 w)
(w 0 w w 0 w 0 w)
(w 0 w 0 0 w 0 w)
(w 0 w 0 0 w 0 w)
(w 0 w w w w 0 w)
(w 0 0 0 0 0 0 w)
(w 0 0 0 0 0 0 w)
(w w w w w w w w)
)
)

(define s3-3
'(
(w w w w w w w w)
(w 1 1 1 1 1 1 w)
(w 1 1 1 1 1 1 w)
(w 1 w w 9 w 1 w)
(w 1 w 1 1 w 1 w)
(w 1 w 1 1 w 1 w)
(w 1 w w w w 1 w)
(w 1 1 1 1 1 1 w)
(w 1 1 1 1 1 1 w)
(w 1 1 1 1 1 1 w)
(w 1 1 1 1 1 1 w)
(w w w w 1 w 1 w)
(w 1 w 1 1 w 1 w)
(w 1 w 1 1 w 1 w)
(w 1 w w w w 1 w)
(w 1 1 w 1 1 1 w)
(w 1 1 1 1 w 1 w)
(w w w w w w w w)
)
)

(define t3-3
'(
(w w w w w w w w)
(w 1 1 1 1 1 1 w)
(w 1 1 1 1 1 1 w)
(w 1 w w 9 w 1 w)
(w 1 w 1 1 w 1 w)
(w 1 w 1 1 w 1 w)
(w 1 w w w w 1 w)
(w 1 1 1 1 1 1 w)
(w 1 1 1 1 1 1 w)
(w 1 1 1 1 1 1 w)
(w 1 1 1 1 1 1 w)
(w w w w 1 w 1 w)
(w 1 w 1 1 w 1 w)
(w 1 w 1 1 w 1 w)
(w 1 w w w w 1 w)
(w 1 1 w 1 1 1 w)
(w 1 1 1 1 w 1 w)
(w w w w w w w w)
)
)

(define s4-3
'(
(w w w w w w w w w w w)
(w 0 0 0 0 0 0 0 0 0 w)
(w w w w w 0 w w w 0 w)
(w w w w w 0 w w w 0 w)
(w 0 0 0 0 0 0 0 w 0 w)
(w 0 0 0 0 0 w 0 w 0 w)
(w w 0 w w 0 w 0 w 0 w)
(w w 0 w 0 0 w 0 w 0 w)
(w w 0 w w w w 0 w 0 w)
(w w 0 0 0 0 0 0 w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w 0 0 0 0 0 0 0 0 0 w)
(w w w w w w w w w w w)
)
)

(define t4-3
'(
(w w w w w w w w w w w)
(w 0 0 0 0 0 0 0 0 0 w)
(w w w w w 0 w w w 0 w)
(w w w w w 0 w w w 0 w)
(w 0 0 0 0 0 0 0 w 0 w)
(w 0 0 0 0 0 w 0 w 0 w)
(w w 0 w w 0 w 0 w 0 w)
(w w 0 w 0 0 w 0 w 0 w)
(w w 0 w w w w 0 w 0 w)
(w w 0 0 0 0 0 0 w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w 0 0 0 0 0 0 0 0 0 w)
(w w w w w w w w w w w)
)
)


(define p0-3
  '(
    (procedure start
      (if mark?
          (get-mark step start turn-180 step turn-180)
          (put-mark)
      )
     )
    (procedure turn-180
      (turn-left turn-left)
    )
    )
)


(define p1-3
  '(
    (procedure start
      put-mark
    )
   )
)

(define p2-3
  '(
    (procedure start
      (if wall? put-mark step)
    )
   )
)

(define p3-3
  '(
    (procedure start
      (if wall? put-mark (step start))
    )
   )
)

(define p4-3
  '(
    (procedure start
      (if wall? put-mark (step start turn-left turn-left step turn-left turn-left))
    )
   )
)

(define p5-3
  '(
    (procedure start
      (if wall? (turn-left start turn-left turn-left turn-left) go-and-return)
    )
    (procedure go-and-return
      (if wall? put-mark (step go-and-return turn-left turn-left step turn-left turn-left))
    )
   )
)

(define p6-3
  '(
    (procedure turn-right (turn-left turn-left turn-left))
    (procedure start
      (if wall? (turn-left (if wall? (turn-left (if wall? (turn-left go-and-return turn-right) go-and-return) turn-right) go-and-return) turn-right) go-and-return)
    )
    (procedure go-and-return
      (if wall? put-mark (step go-and-return turn-left turn-left step turn-left turn-left))
    )
   )
)

(define p7-3
  '(
    (procedure start fill-maze)
    (procedure fill-maze
      (if mark?
          ()
          ( put-mark
            (if wall? () (step fill-maze step-back))
            turn-left
            (if wall? () (step fill-maze step-back))
            turn-left
            turn-left
            (if wall? () (step fill-maze step-back))
            turn-left
          ) 
     ))
    (procedure step-back
      (turn-left turn-left step turn-left turn-left)
    )
    )
)

(define p8-3
  '(
    (procedure start add-mark-to-maze)
    (procedure add-mark-to-maze
      (if mark?
       (get-mark
        (if mark?
          (put-mark)
          ( put-mark put-mark
            (if wall? () (step add-mark-to-maze step-back))
            turn-left
            (if wall? () (step add-mark-to-maze step-back))
            turn-left
            turn-left
            (if wall? () (step add-mark-to-maze step-back))
            turn-left get-mark
          ))
       ) (put-mark add-mark-to-maze get-mark)
     ))
    (procedure step-back
      (turn-left turn-left step turn-left turn-left)
    )
    )
)

(define p9-3
  '(
     (procedure start () )
   )
)

(define p10-3
  '(
    (procedure start (go-north go))
    (procedure go
      (if wall?
          (turn-left go)
          (step go-north go)
      )
     )
    (procedure go-north
      (if north? () (turn-left go-north))
     )
    )
)

(define p11-3
  '(
    (procedure start (turn-north go))
    (procedure go (if wall? () (step go)))
    (procedure turn-north (if north? () (turn-left turn-north)))))


(newline)



;PUBLIC 4
(newline)
(display "PUB04")

(define s0-4
'(
(w w w)
(w 0 w)
(w w w)
)
)

(define t0-4
'(
(w w w)
(w 0 w)
(w w w)
)
)

(define s1-4
'(
(w w w)
(w 1 w)
(w 0 w)
(w w w)
)
)

(define t1-4
'(
(w w w)
(w 1 w)
(w 0 w)
(w w w)
)
)

(define s2-4
'(
(w w w w w w)
(w 0 0 1 1 w)
(w 0 0 0 1 w)
(w 0 0 0 1 w)
(w 0 0 1 0 w)
(w 0 1 0 1 w)
(w w w w w w)
)
)

(define t2-4
'(
(w w w w w w)
(w 0 0 1 1 w)
(w 0 0 0 1 w)
(w 0 0 0 1 w)
(w 0 0 1 0 w)
(w 0 1 0 1 w)
(w w w w w w)
)
)

(define s3-4
'(
(w w w w w w w w w w w)
(w 0 0 0 0 0 0 0 0 0 w)
(w w w w w 0 w w w 0 w)
(w w w w w 0 w w w 0 w)
(w 0 0 0 0 0 0 0 w 0 w)
(w 0 0 0 0 0 w 0 w 0 w)
(w w 0 w w 0 w 0 w 0 w)
(w w 0 w 0 0 w 0 w 0 w)
(w w 0 w w w w 0 w 0 w)
(w w 0 0 0 0 0 0 w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w 0 0 0 0 0 0 0 0 0 w)
(w w w w w w w w w w w)
)
)

(define t3-4
'(
(w w w w w w w w w w w)
(w 0 0 0 0 0 0 0 0 0 w)
(w w w w w 0 w w w 0 w)
(w w w w w 0 w w w 0 w)
(w 0 0 0 0 0 0 0 w 0 w)
(w 0 0 0 0 0 w 0 w 0 w)
(w w 0 w w 0 w 0 w 0 w)
(w w 0 w 0 0 w 0 w 0 w)
(w w 0 w w w w 0 w 0 w)
(w w 0 0 0 0 0 0 w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w w w w w w w w w 0 w)
(w 0 0 0 0 0 0 0 0 0 w)
(w w w w w w w w w w w)
)
)

(define p0-4
   '( 
      (procedure start
         (turn-left (if wall? (turn-right 
             (if wall? (turn-right (if wall? turn-right step)) step)) step)
                 put-mark start )
      )   
      (procedure turn-right (turn-left turn-left turn-left))
  )
)


(define p1-4
  '(
    (procedure start
      put-mark
    )
   )
)

(define p2-4
  '(
    (procedure start
      (if wall? put-mark step)
    )
   )
)


(define p7-4
  '(
    (procedure start add)
    (procedure add
      (sub-one turn-180 go turn-right step turn-right 
       add-one turn-180 go turn-left  step turn-left 
       add
      )
    )
    (procedure add-one
      (if mark?
          (get-mark (if wall? (turn-180 go turn-180) (step add-one)))
          (put-mark)
      )
     )
    (procedure sub-one
      (if mark?
          (get-mark)
          (put-mark (if wall? (turn-180 go turn-right step (if wall? turn-right (turn-right sub-one))) (step sub-one)))
      )
     )
    (procedure turn-180
      (turn-left turn-left)
     )
    (procedure turn-right
      (turn-left turn-left turn-left)
     )
    (procedure go
      (if wall? () (step go))
    )
  )
)

(define p8-4
  '(
    (procedure start (go-north go))
    (procedure go
      (if wall?
          (turn-left go)
          (step go-north go)
      )
     )
    (procedure go-north
      (if north? () (turn-left go-north))
     )
    )
)

(define p9-4
  '(
    (procedure start (turn-north go))
    (procedure go
      (if wall?
          ()
          (step go)
      )
     )
    (procedure turn-north
      (if north? () (turn-left turn-north))
     )
    )
)

(newline)
;(simulate `(,s3-4 (1 1) north) 'start p6-4 30 100)

(assert-equal "pub3" (evaluate `(,p0-4 ,p1-4 ,p2-4 ,p7-4 ,p8-4 ,p9-4)
          `(((,s0-4 (1 1) west) 
             (,t0-4 (1 1) north)) 
             ((,s1-4 (1 2) west) 
             (,t1-4 (1 1) north)) 
             ((,s2-4 (4 5) west) 
             (,t2-4 (4 1) north)) 
             ((,s3-4 (2 9) east) 
             (,t3-4 (2 4) north))) `(20 8 20 50) 5) '(
((0 1 11 19) ((procedure start (turn-north go)) (procedure go (if wall? () (step go))) (procedure turn-north (if north? () (turn-left turn-north)))))
((0 3 14 26) ((procedure start (go-north go)) (procedure go (if wall? (turn-left go) (step go-north go))) (procedure go-north (if north? () (turn-left go-north)))))
))




;PUBLIC 5
(newline)
(display "PUB05")



;PUBLIC 6
(newline)
(display "PUB06")



;PUBLIC 7
(newline)
(display "PUB07")
