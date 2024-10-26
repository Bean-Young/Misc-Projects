DATA SEGMENT
x db 1,2,3,4,5,6,7,8,9,'ABCDEFG'
Y db 16 dup (0)
DATA ENDS

CODE SEGMENT
assume cs:CODE, ds:DATA
Start:
    mov ax, DATA    
    mov ds, ax     
    
    mov bx, offset x 
    mov bp, offset Y 
    mov cx, 10h      
	
copy:
    mov al, [bx]    
    mov [bp], al    
    inc bx          
    inc bp          
    loop copy
	mov ah, 09h         
    mov dx, offset Y       
    int 21h        
    mov ah, 4ch   
    int 21h
CODE ENDS

END Start
