data segment
    scores db 59, 62, 70, 48, 91, 85, 66, 55, 75   
    count  db 0            
data ends

code segment
assume cs:code, ds:data

start:
    mov  ax, data
    mov  ds, ax

    lea  bx, scores          
    mov  cx, 9

count_loop:
	mov al,[bx]
    cmp al, 60   
    jb next_score    
    inc  count
	
next_score:
    inc  bx                 
    loop count_loop          

    mov  ah, 02h             
    mov  dl, count         
    add  dl, '0'            
    int  21h                

    mov  ax, 4c00h
    int  21h

code ends
end start
