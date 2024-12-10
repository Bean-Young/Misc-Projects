Data segment
	num db 0AAH,01H,02H,0ABH,03H,04H,05H,06H,07H,0FFH
	count dw 10
Data ends
 
Code segment
	assume cs:Code, ds:Data

start:
    mov ax, Data
    mov ds, ax

    mov cx, count
    mov si, 0

positive:
    mov al, [num + si]
    test al, 80H
    jnz skip

    mov ah, 0Eh
    add al, '0'
    int 10h

skip:
    inc si
    loop positive

    mov ah, 4Ch
    int 21h

Code ends
	end start
