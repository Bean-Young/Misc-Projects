Code  segment
Assume  cs:code
 Mov  AX, 1234h
 Xchg  ah, al
MOV  ah, 4ch
 Int  21h
Code  ends
  end
