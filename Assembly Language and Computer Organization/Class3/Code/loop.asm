data segment
data ends
code segment
assume cs:code,ds:data 
start:
    MOV AX, 0         
    MOV DS, AX        
    MOV BX, 0200H      
    MOV CX, 40H
    XOR DI, DI         

A:
    MOV [BX], DI       
    INC DI             
    INC BX            
    LOOP A    
	MOV AH,4CH
	INT 21H
code ends
	end start