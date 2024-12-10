DATA	SEGMENT
	org 12h
	db1  db 10h, 23h
	org $+30h
	var1  dw  $+8
	x  db 'AAA'
DATA	ENDS

Code  segment
    assume   cs:code,ds:data
Start:  mov   ax,  data
        mov   ds, ax
		    mov  bx, offset  db1
        mov  bp, offset  var1
		    mov  dx, var1
		    mov  ah, 4ch
        int   21h
CODE  	ENDS
	END	START