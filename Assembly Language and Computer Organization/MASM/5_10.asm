CODE		SEGMENT
    assume   cs:code 
 exreg   macro  op
    push  ax        ; ax入栈保存
    mov  ax,   op
    xchg  ah, al     ; 高8、低8位互换
    mov  op, ax     ; 重新赋值
    pop  ax         ; 恢复ax
  endm

START:  MOV  BX, 1234H
		EXREG  BX    ; 测试，执行后 (bx)=3412h
		MOV   CX, 5566H
		EXREG  CX    ; 测试，执行后 (CX)=6655h
	
		mov  ah, 4ch
        int   21h
code  ends
 end  START