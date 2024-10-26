Datas segment
	decnum db 5 dup(?)
Datas ends

Code segment
	assume cs:Code, ds:Datas

start:
    mov ax, Datas
    mov ds, ax
	mov bx, 10111100B
    mov ax, bx
    mov cx, 10
    mov si, 4

convert_loop:
    xor dx, dx
    div cx
    add dl, '0'
    mov [decnum + si], dl
    dec si
    cmp ax, 0
    jne convert_loop

    mov si, 0

print_loop:
    mov ah, 0Eh
    mov al, [decnum + si]
    int 10h
    inc si
    cmp si, 5
    jne print_loop

    mov ah, 4Ch
    int 21h

Code ends
	end start
