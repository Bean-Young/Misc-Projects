DATA	SEGMENT
  num1  equ 10h
  num2  equ  num1 mod  10h
  num3  db  (12 or 6 and 2)   le 0eh  ;  FFH
  num4  db  num1 dup(?)             ;  10H 个0
  num5  dw  num3         ;  取num3的偏移量, 值为0000H
  x  db 'AAA'
DATA	ENDS

Code  segment
    assume   cs:code,ds:data
Start:  mov   ax,  data
        mov   ds, ax
		    mov  ah, 4ch
        int   21h
CODE  	ENDS
	END	START