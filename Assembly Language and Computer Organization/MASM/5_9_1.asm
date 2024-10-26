DATA	SEGMENT
  org 0400h
  arword equ this word
  arbyte  db  50 dup(0)
DATA	ENDS
CODE SEGMENT
    ASSUME CS:CODE, DS:DATA
START:
	MOV AH, 4CH
    INT 21H
CODE ENDS

END START
