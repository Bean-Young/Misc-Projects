Code  segment
Assume  cs:code
 Mov  AX, 1234h
 Mov  CL, 8
 ROL   ax, CL      
 MOV  ah, 4ch
 Int  21h
Code  ends
 End