mystack 	SEGMENT   para  stack
       db  50 dup(0)
       top  label word  ; 标记栈底位置
mystack	ENDS
CODE		SEGMENT
    assume   cs:code,   ss:mystack
		MOV	AX , mySTACK 	    ;设置SS
		MOV	SS , AX
		MOV	SP , OFFSET   TOP	;设置SP
code  ends
 end
 CODE SEGMENT
    ASSUME CS:CODE, DS:DATA
START:
	MOV AH, 4CH
    INT 21H
CODE ENDS

END START
