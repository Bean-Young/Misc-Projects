data segment
data ends
code segment
assume cs:code,ds:data 
start:
    MOV AX, 0       
    MOV DS, AX          

    MOV CX, 0AH       
    MOV DL, 30H         
A:
    MOV AH, 02H         
    INT 21H             
    INC DL             
    LOOP A
	
	
	MOV CX, 1AH         
    MOV DL, 41H
B:
    MOV AH, 02H         
    INT 21H            

    INC DL              
    LOOP B      
	
	MOV CX,1AH
	MOV DL,61H
C:
    MOV AH, 02H
    INT 21H
    INC DL
    LOOP C

    MOV AH, 4CH         
    INT 21H  
code ends
	end start