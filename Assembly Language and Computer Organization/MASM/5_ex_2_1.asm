Code  segment
Assume  cs:code
 Mov  AX, 1234h
 Mov  CL, 8
 ROL   ax, CL      ;循环移位8次即可. 掌握ROL指令
 MOV  ah, 4ch
 Int  21h
Code  ends
  End