data segment
    BUF  DB  5, 6, 7, 58H, 62, 45H, 127
    COUNT EQU $-BUF
	MAX  DB  ?
data ends

code segment
assume cs:code, ds:data

start:
    mov  ax, data
    mov  ds, ax

    mov  cx, COUNT
    mov  si, OFFSET BUF
    mov  al, [si]
    inc  si
    dec  cx

find_max:
    cmp  cx, 0
    je   get_max
    mov  bl, [si]
    inc  si
    dec  cx
    cmp  al, bl
    ja   find_max
    mov  al, bl
    jmp  find_max

get_max:
    mov  MAX, al

    mov  ax, 4C00h
    int  21h

code ends
end start
