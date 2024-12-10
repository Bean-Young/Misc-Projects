data segment
    sum dw 0
data ends

code segment
assume cs:code, ds:data

start:
    mov  ax, data
    mov  ds, ax

    mov  cx, 50       
    mov  bx, 2        

loop_sum:
    add  sum, bx     
    add  bx, 2        
    loop loop_sum    
	
    mov  dx, sum
    mov  ax, 4C00h
    int  21h

code ends
end start
