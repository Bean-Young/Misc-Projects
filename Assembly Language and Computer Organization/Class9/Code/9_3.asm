data segment
data ends

code segment
assume cs:code, ds:data

start:
    mov  ah, 01h
    int  21h
    cmp  al, 'y'
    je   YES
    cmp  al, 'n'
    je   NO
    jmp  start

YES:
	mov  ax, 0000h
    mov  ax, 4C00h
    int  21h

NO:
	mov  ax, 0FFFFh
    mov  ax, 4C00h
    int  21h

code ends
end start
