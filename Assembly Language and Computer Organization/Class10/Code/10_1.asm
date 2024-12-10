data segment
    ary1 db 12,-35,0,126,-90,-5,68,120,1,-19
    ary2 db 24,25,0,-38,-89,99,68,100,2,-20
data ends

code segment
assume cs:code, ds:data

start:
    mov  ax, data
    mov  ds, ax

    mov  cx, 10      
    mov  si, 0        
    mov  di, 0       
	
compare:
    mov  al, ary1[si]
    mov  bl, ary2[di]
    
    cmp  al, bl       
    jg   In_ary1
    
    mov  ary1[si], bl 
    mov  ary2[di], al 
    jmp  next

In_ary1:
    mov  ary1[si], al 
    mov  ary2[di], bl

next:
    inc  si          
    inc  di
    loop compare
	
    mov  ax, 4C00h
    int  21h

code ends
end start
