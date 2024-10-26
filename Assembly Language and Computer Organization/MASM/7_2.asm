data segment 
    A   db 'abcde'      
    B   db 'abxde'     
    FLG db 0            
    yes db 'equal$'    
    no  db 'not equal$'
data ends

code segment
assume cs:code, ds:data

start:
    mov ax, data
    mov ds, ax

    lea si, A          
    lea di, B           
    mov cx, 5           
    repe cmpsb          

    jz equal         
    mov FLG, 0          
    lea dx, no          
    jmp exit

equal:
    mov FLG, 1         
    lea dx, yes         

exit:
    mov ah, 9
    int 21h             

    mov ah, 4ch        
    int 21h             
code ends
end start
