DATA SEGMENT
    ORG 0B000H   

    DATA_ARRAY DB 100 DUP(2)  

DATA ENDS

CODE SEGMENT
    ASSUME CS:CODE, DS:DATA
START:
	MOV AX,076AH
	MOV DS,AX
	MOV BH,[DATA_ARRAY]
	MOV CX,[DATA_ARRAY]
	MOV AH, 4CH
    INT 21H
CODE ENDS

END START
