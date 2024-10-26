DATA	SEGMENT
  org 30h
  num=20h
  da1  dw 10h, $+20h, 20h, $+30h
  da2  dw  da1+num+10h 
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