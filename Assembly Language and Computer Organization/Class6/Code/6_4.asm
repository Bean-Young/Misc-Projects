DATA SEGMENT
input db 'Enter a lowercase letter: $'
output db 0DH, 0AH, 'The uppercase letter is: $'
DATA ENDS

CODE SEGMENT
assume cs:CODE, ds:DATA
Start:
    mov ax, DATA    
    mov ds, ax     
    
input_loop:

    mov ah, 09h         
    mov dx, offset input    
    int 21h           
    
 
    mov ah, 01h        
    int 21h            

    cmp al, '!'         
    je exit_program     
    
    cmp al, 'a'         
    jl input_loop      
    cmp al, 'z'         
    jg input_loop       
    sub al, 20h   
	

    mov ah, 09h         
    mov dx, offset output    
    int 21h  
	
	mov dl,al
    mov ah, 02h        
    int 21h             
	
    mov ah, 02h        
    mov dl, 0DH         
    int 21h             
    mov dl, 0AH         
    int 21h             

    jmp input_loop      

exit_program:
    mov ah, 4ch         
    int 21h
CODE ENDS

END Start
