data segment
    A db 'AAAAA$'
    B db 'BBBBB$'
    C db 'CCCCC$'
    temp db 5 dup(?) 
data ends

code segment
assume cs:code, ds:data

start:
    mov  ax, data
    mov  ds, ax
	mov  es, ax 
    cld              
	
    lea  si, A   
    lea  di, temp  
    mov  cx, 5 
    repe movsb 
	
    lea  si, C 
    lea  di, A     
    mov  cx, 5   
    repe movsb  

    lea  si, temp  
    lea  di, C  
    mov  cx, 5     
    repe movsb      ;C A change
		
    lea  si, C   
    lea  di, temp 
    mov  cx, 5   
    repe movsb    

    lea  si, B   
    lea  di, C     
    mov  cx, 5     
    repe movsb     

    lea  si, temp  
    lea  di, B     
    mov  cx, 5    
    repe movsb     ;C B change
	
	lea dx,A
	mov ah,09
	int 21h
	
	lea dx,B
	mov ah,09
	int 21h
    
	lea dx,C
	mov ah,09
	int 21h
	
	mov  ax, 4c00h
    int  21h

code ends
end start
