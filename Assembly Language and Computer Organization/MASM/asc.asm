data segment
data ends
code segment
assume cs:code,ds:data 
start:
    MOV AX, 0       
    MOV DS, AX          

    MOV CX, 1AH       
    MOV DL, 41H         

A:
    MOV AH, 02H         
    INT 21H            

    INC DL              
    LOOP A      

    MOV AH, 4CH         
    INT 21H  
code ends
	end start