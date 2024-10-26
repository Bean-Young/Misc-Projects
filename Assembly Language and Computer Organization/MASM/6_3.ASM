DATA SEGMENT
count db 0
str db 20h,20h,'abc def',20h,90 dup (0)
DATA ENDS

CODE SEGMENT
assume cs:CODE, ds:DATA
Start:
    mov ax, DATA    
    mov ds, ax     
    
    mov bp, offset str 
    mov cx, 100     
    mov bl, 20h     
    mov count,0
    
count_loop:
    mov al, [bp]   
    cmp al, bl      
    jnz not_space 
    inc count      
not_space:
    inc bp         
    loop count_loop 

	MOV  ah, 4ch
	Int  21h
CODE ENDS

END Start

