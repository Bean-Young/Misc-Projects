DATA SEGMENT
    RES DB ?                
    BUF DB 2,3,9,-4,8,6,0,3,-3,-4,8,7,9,2,3,5,-9,4,5,8
    COUNT EQU 20          
DATA ENDS

CODE SEGMENT
    ASSUME CS:CODE,DS:DATA

start:
    mov ax, data
    mov ds, ax             
    mov cx, count
    mov si, 0              

negative:
    mov al, BUF[si]        
    cmp al, 0             
    jl result 
    inc si                 
    loop negative    
result:
    mov RES, al           
    mov ax, 4C00H         
    int 21H

CODE ENDS
END START
