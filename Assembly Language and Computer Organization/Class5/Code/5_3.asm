data  segment
   org  20h      ; 规定起始位置
   var1  db  20h  dup(0)
   var2  dw  30h  dup(0)
   var3  dw  12h  dup(4 dup(2), 30h)
data  ends

code  segment
assume  cs:code,  ds:data
start:   mov  ax, data
        mov  ds, ax
		    mov   AL,   LENGTH  VAR1  ;  H
        MOV   AH,  SIZE      VAR1  ; H
        MOV   BL,   LENGTH  VAR2  ;  H
        MOV   BH,  SIZE     VAR2   ;  H
        MOV   CL,   LENGTH  VAR3  ; H
        MOV   CH,   SIZE  VAR3     ; H
        mov   ah, 4ch
        int  21h
 code  ends
end  start