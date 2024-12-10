data segment
data ends

code segment
assume cs:code, ds:data

start:
    mov  ax, data
    mov  ds, ax

    mov  ah, 0AH;1010
    mov  al, 04H;0010;0100

    mov  bl, ah
    mov  bh, al
    and  bl, 4
    and  bh, 4

    cmp  bl, bh
    je   equal

    mov  ah, 0FFh
    jmp  unequal

equal:
    mov  ah, 0

unequal:
    mov  ax, 4C00h
    int  21h

code ends
end start
