Datas segment
	STU db 76,69,84,90,73,88,99,63,100,80
	S6 db 0
	S7 db 0
	S8 db 0
	S9 db 0
	S10 db 0
Datas ends 
 
Code segment
	assume ds:Datas,cs:Code

start:
    mov ax, Datas
    mov ds, ax

    mov cx, 10
    mov si, 0

scores:
    mov al, [STU + si]
    cmp al, 60
    jl next

    cmp al, 70
    jl s6_case
    cmp al, 80
    jl s7_case
    cmp al, 90
    jl s8_case
    cmp al, 100
    jl s9_case
    je s10_case

s6_case:
    inc S6
    jmp next

s7_case:
    inc S7
    jmp next

s8_case:
    inc S8
    jmp next

s9_case:
    inc S9
    jmp next

s10_case:
    inc S10
    jmp next

next:
    inc si
    loop scores

    mov ah, 4Ch
    int 21h

Code ends
	end start
